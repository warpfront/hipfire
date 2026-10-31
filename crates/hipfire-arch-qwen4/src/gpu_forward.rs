// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP Qwen4 forward execution.
//!
//! This module is the production device path.  The CPU/reference equations in
//! reference_forward are intentionally separate and are never called
//! here.  All
//! learned projections consume the resident tensor supplied by the assembled
//! bundle; MQv2 projections are rotated and dispatched through their native
//! qt44/qt53 kernels, while recurrent/cache state remains resident on the GPU.
//!
//! MoE routing has two sealed families.  QT44/QT53 layers seal the indexed
//! decode route for one routing row and the grouped prefill route above it.
//! The stock-FP8 recipe's layers (gfx942) bind the qt=55 `FP8E4M3B128` expert
//! blocks of the loader's immutable accessor tables into the same expert
//! runtime ([`Qwen4MoeLayerRuntime`]: per-expert natural-basis `WeightRef`s,
//! contiguous base, block stride) and declare
//! [`MoeRouteCapability::Fp8B128Grouped`], which always seals the grouped
//! prefill route, one row included (`program::seal_moe_for_layer`).

use crate::bundle::Qwen4Bundle;
use crate::config::{LayerType, Qwen4Config, Qwen4FfnConfig};
use crate::ep_seal::Qwen4EpRankSeal;
use crate::ple_stage::{
    ple_async_upload, ple_async_upload_enabled, PleHostStage, PLE_ASYNC_UPLOAD_ENV,
};
use crate::program::{
    execute_final_hyper, execute_lm_head, validate_final_hyper, validate_lm_head,
    Qwen4AttentionWeights, Qwen4DenseBinding, Qwen4FfnBinding, Qwen4FfnDims, Qwen4GdnWeights,
    Qwen4HyperReadWeights, Qwen4HyperWeights, Qwen4LayerDescription, Qwen4LayerScratch,
    Qwen4MoeBinding, Qwen4PleWeights, Qwen4ProgramDims, Qwen4QsaWeights, Qwen4ScratchLayout,
    QWEN4_QSA_KERNEL_ROPE_THETA,
};
use crate::projection::{row_stride, ProjectionView};
use crate::weights::{
    Fp8MoeExpertTables, HyperConnectionReadWeights, HyperConnectionWeights, MoeWeights,
    Qwen4BandScope, Qwen4EpPlacement, Qwen4FfnWeights, Qwen4LayerWeights, Qwen4Weights, TensorRef,
    WeightError,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoeRecipe, MoeRouteCapability,
    MoeRouteFormats, MoeRouteGeometry, MoeRoutePolicy, MoeRowGather, MoeSharedDecode,
    MoeSharedDtypes, MoeSharedWeights, MoeStageTables, MoeTileExchange, MoeTileExchangeParams,
    RoutedExpertWeights, MOE_GROUPED_BLOCK_M,
};
use hipfire_dispatch::pipeline::gated_mlp::{
    Activation, GatedMlpOp, GatedMlpScratch, MlpLadder, MlpOutput, MlpStorage,
};
use hipfire_dispatch::pipeline::sealed_moe::{
    retained_body_action, specialized_sealed_moe_retained_admission, RetainedBodyAction,
    SealedMoeCall,
};
use hipfire_dispatch::pipeline::{
    execute_embedding, execute_validated_steps, execute_validated_steps_with_moe_hooks,
    seal_decode, validate_steps, BoundMoeExperts, ClearOp, EmbeddingOp, ExpertBindingCache,
    ExpertMetadata, ExpertResource, ExpertResources, ExpertTable, GatedDeltaNetOp,
    GroupedDepthwiseOp, HyperReadOp, HyperWriteOp, IndexedAttentionMode, IndexedAttentionOp,
    IndexedAttentionState, MixerPhase, MoeStepHooks, Step,
};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::external_rows::{RowFetch, RowTicket};
use hipfire_runtime::multi_gpu::executor::MeshRoute;
use hipfire_runtime::weight_manifest::ExpertSourceLayout;
use rdna_compute::moe::Qwen4EpOwnerTables;
use rdna_compute::replay::{ReplayState, ShadowBodyRoute};
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor, TrunkFamily};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

#[cfg(target_os = "linux")]
#[path = "npu_trunk/forward.rs"]
mod npu_forward;

pub(crate) const EPSILON: f32 = 1.0e-6;
const GFX942_PREFILL_CHUNK_ROWS: usize = 1536;

/// Developer override of the requested prefill chunk
/// ([`qwen4_prefill_chunk_requested`]).
pub(crate) const QWEN4_PREFILL_CHUNK_ROWS_ENV: &str = "HIPFIRE_PREFILL_CHUNK_ROWS";

/// Default rows of one Qwen4 prefill chunk (the reusable forward scratch).
///
/// Public serving calls may receive longer prompts; the forward owner tiles
/// those requests over this bounded capacity instead of allocating
/// prompt-sized grouped MoE buffers.  Every chunk re-streams all routed
/// expert weights, so pp throughput rises with the chunk until another
/// kernel's shape envelope ends.  Measured against the earlier 1536-row
/// chunk: gfx1151 pp8192 +10.2% at 8192 rows (resident experts).  On
/// gfx1201 the default depends on the forward scratch plan `mode`: with the
/// compact phase-arena scratch ([`Qwen4ForwardScratchMode::FnSymGfx1201`])
/// the chunk is 16384 rows (pp16384 +12% over 4096 rows measured on the
/// R9700, fn-chunk-ab-001; the scratch costs 4776 MiB at 16384 rows against
/// 4328 MiB for 4096 rows under the Full plan).  Fewer routed experts stay
/// resident than with a 4096-row compact chunk, so decode is slower; an
/// explicit `HIPFIRE_PREFILL_CHUNK_ROWS=4096` keeps the smaller chunk for
/// decode-heavy use.  Under the Full plan the
/// larger chunk costs VRAM expert layers under `auto` expert placement, so
/// the rung is the fastest pp8192 within 3% of the 2048-row rung's tg64:
/// 4096 rows (14 layers at max_seq 66,560) measured pp8192 +14.1% over 2048
/// rows (16 layers) at -2.9% tg64; 8192 rows (11 layers) measured +22.6% at
/// -6.9% tg64.
///
/// gfx1100 takes 8192 only on a measured narrow discrete host link
/// ([`crate::host_link::HostLink::is_narrow_discrete`], measured below
/// 16 GB/s; the same predicate as the G2 default,
/// [`crate::host_link::expert_stage_default`]): every chunk re-streams the
/// host experts through the narrow link, so the chunk's row count divides
/// that cost, and G2 staging hides the copy behind the chunk's compute. The
/// expert placement is sized from this same chunk
/// ([`qwen4_prefill_chunk_requested`]), so the 8192-row reserve cedes
/// resident expert layers there; fast, unmeasured and unified links keep
/// 1536. `link` is only evaluated for gfx1100, so no other arch pays for the
/// probe here. Other architectures keep the measured 1536.
pub fn qwen4_prefill_chunk_default(
    arch: &str,
    mode: Qwen4ForwardScratchMode,
    link: impl FnOnce() -> crate::host_link::HostLink,
) -> usize {
    match rdna_compute::expert_capabilities::narrow_link_prefill_chunk_rows(arch) {
        Some(rows) if link().is_narrow_discrete() => rows,
        _ => qwen4_prefill_chunk_baseline_default(arch, mode),
    }
}

/// The default chunk before any link policy: gfx1151 8192, gfx1201 16384
/// under the compact [`Qwen4ForwardScratchMode::FnSymGfx1201`] scratch and
/// 4096 under `Full`, gfx942 and every other arch 1536. Never reads the host
/// link; gfx1100 uses it unless [`qwen4_prefill_chunk_default`] sees a narrow
/// discrete link.
pub fn qwen4_prefill_chunk_baseline_default(arch: &str, mode: Qwen4ForwardScratchMode) -> usize {
    match arch {
        "gfx1151" => 8192,
        "gfx1201" if mode == Qwen4ForwardScratchMode::FnSymGfx1201 => 16384,
        "gfx1201" => 4096,
        "gfx942" => GFX942_PREFILL_CHUNK_ROWS,
        _ => 1536,
    }
}

/// Requested prefill chunk: `prefill.chunk_rows`
/// (`HIPFIRE_PREFILL_CHUNK_ROWS`) rounded down to whole 256-row GEMM tiles,
/// else [`qwen4_prefill_chunk_default`] for `gpu` under scratch `mode` (its
/// host link is measured once per device, see [`crate::host_link`], and only
/// on gfx1100); never above `max_seq`. An explicit value never probes.
/// Load-time admission ([`qwen4_prefill_chunk_rungs`]) may still pick a
/// smaller rung.
///
/// This is also the chunk the loader sizes the expert placement reserve,
/// forward census and pool workspace from, so on a measured narrow discrete
/// gfx1100 link the default 8192 rows are charged to the placement (ceding
/// resident expert layers), and an explicit `HIPFIRE_PREFILL_CHUNK_ROWS`
/// stays authoritative for placement and chunk alike.
pub fn qwen4_prefill_chunk_requested(
    gpu: &Gpu,
    max_seq: usize,
    mode: Qwen4ForwardScratchMode,
) -> usize {
    prefill_chunk_requested_from(
        &gpu.arch,
        hipfire_config::developer_var(QWEN4_PREFILL_CHUNK_ROWS_ENV)
            .ok()
            .as_deref(),
        max_seq,
        mode,
        || crate::host_link::host_link(gpu),
    )
}

/// The explicit `HIPFIRE_PREFILL_CHUNK_ROWS` value rounded down to whole
/// 256-row tiles; `None` when unset, unparsable or below one tile.
pub(crate) fn explicit_prefill_chunk_rows(env: Option<&str>) -> Option<usize> {
    env.and_then(|value| value.trim().parse::<usize>().ok())
        .map(|rows| rows / 256 * 256)
        .filter(|&rows| rows >= 256)
}

/// [`qwen4_prefill_chunk_requested`] with the `HIPFIRE_PREFILL_CHUNK_ROWS`
/// value and the (lazily measured) host link supplied.
pub(crate) fn prefill_chunk_requested_from(
    arch: &str,
    env: Option<&str>,
    max_seq: usize,
    mode: Qwen4ForwardScratchMode,
    link: impl FnOnce() -> crate::host_link::HostLink,
) -> usize {
    explicit_prefill_chunk_rows(env)
        .unwrap_or_else(|| qwen4_prefill_chunk_default(arch, mode, link))
        .min(max_seq)
}

/// Flash-Next pipeline-parallel prefill chunk on gfx1201. Under PP the band
/// pipeline needs several chunks in flight; the 16384-row compact default
/// leaves a 32K prompt only 2-3 chunks, so smaller tiles fill it. Measured
/// on 4x R9700 (PP4, packed mq4-xts, fresh processes, greedy), prefill tok/s
/// at 16384 / 8192 / 4096 rows: 8K 4,838 / 6,039 / 6,709; 32K 6,953 / 14,026 /
/// 16,657; 64K 13,445 / 16,735 / 18,260; 131K 15,194 / 17,444 / 17,885.
/// Decode is unchanged (~54 tok/s). Numerics equal a single card at the same
/// explicit chunk (`HIPFIRE_PREFILL_CHUNK_ROWS=4096`), which is also the
/// gfx1201 Full-plan / EP default.
pub const QWEN4_PP_PREFILL_CHUNK_ROWS_GFX1201: usize = 4096;

/// [`qwen4_prefill_chunk_requested`] for a pipeline-parallel band or mesh:
/// an explicit `HIPFIRE_PREFILL_CHUNK_ROWS` wins, else
/// [`QWEN4_PP_PREFILL_CHUNK_ROWS_GFX1201`] on gfx1201, else the single-card
/// default; never above `max_seq`. This is the REQUESTED chunk: the loader
/// reads it once, the PP auto-fit (`admission::qwen4_pp_auto_fit`) may step it
/// down to fit the cards, and the chosen chunk is what every band and the mesh
/// then share (a band never re-derives it).
pub fn qwen4_pp_prefill_chunk_requested(
    gpu: &Gpu,
    max_seq: usize,
    mode: Qwen4ForwardScratchMode,
) -> usize {
    pp_prefill_chunk_requested_from(
        &gpu.arch,
        hipfire_config::developer_var(QWEN4_PREFILL_CHUNK_ROWS_ENV)
            .ok()
            .as_deref(),
        max_seq,
        mode,
        || crate::host_link::host_link(gpu),
    )
}

/// Whether the operator pinned the prefill chunk (`HIPFIRE_PREFILL_CHUNK_ROWS`
/// holds a value of at least one 256-row tile, the value
/// [`qwen4_pp_prefill_chunk_requested`] returns first). The PP auto-fit never
/// moves a pinned chunk, only the split.
pub fn qwen4_prefill_chunk_rows_pinned() -> bool {
    explicit_prefill_chunk_rows(
        hipfire_config::developer_var(QWEN4_PREFILL_CHUNK_ROWS_ENV)
            .ok()
            .as_deref(),
    )
    .is_some()
}

/// [`qwen4_pp_prefill_chunk_requested`] with the env value and host link supplied.
pub(crate) fn pp_prefill_chunk_requested_from(
    arch: &str,
    env: Option<&str>,
    max_seq: usize,
    mode: Qwen4ForwardScratchMode,
    link: impl FnOnce() -> crate::host_link::HostLink,
) -> usize {
    explicit_prefill_chunk_rows(env)
        .unwrap_or_else(|| {
            if arch == "gfx1201" {
                QWEN4_PP_PREFILL_CHUNK_ROWS_GFX1201
            } else {
                qwen4_prefill_chunk_default(arch, mode, link)
            }
        })
        .min(max_seq)
}

/// Chunk rungs admission tries, largest first: `requested`, then each
/// smaller default rung (8192, 4096, 2048) down to 1536 (or `requested`
/// itself when smaller).
pub(crate) fn qwen4_prefill_chunk_rungs(requested: usize) -> impl Iterator<Item = usize> {
    std::iter::once(requested).chain(
        [8192, 4096, 2048, 1536]
            .into_iter()
            .filter(move |&rows| rows < requested),
    )
}

/// Physical rows of the speculative logit and argmax buffers. Native MTP
/// reserves one row for its seed; prompt advances and MTP prefill read their
/// final row only. Actual admission also requires allocated forward scratch.
pub(crate) const QWEN4_SPEC_VERIFY_ROWS: usize = hipfire_config::QWEN4_VERIFY_MAX_ROWS;

/// Rows a verify forward on this GPU runs row-exact (each row bitwise the
/// sequential single-row decode): the per-arch value of
/// [`rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows`] (the
/// historic few-row limit 8 until the oracle proves the arch wider, or the
/// `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS` developer override). The single per-GPU
/// value the verify scope and the n-gram copy lane read.
pub(crate) fn qwen4_row_exact_verify_rows(gpu: &Gpu) -> usize {
    rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(&gpu.arch)
}

/// How a plain one-row autoregressive (AR) decode of this process launches
/// its body, as far as arithmetic is concerned.
///
/// Qwen4 production has exactly two modes: ordinary eager HIP launches, and the
/// retained Redline body (`ReplayController`: the first eligible plain-AR
/// forward is *recorded*, later ones replay the PM4/AQL tape of that record).
/// A recorded launch runs with `Gpu::replay.is_recording()`, so every
/// "not recorded / not captured" route (QSA split-K decode attention above all)
/// is closed in it and in its replays. No Qwen4 production path opens a
/// hipGraph capture (`graphs.capture_mode` is set only by oracle examples);
/// it is folded into [`Self::Recorded`] because it closes the same routes.
/// A verify forward (2+ rows) is never a retained candidate
/// (`n == 1` in the `eligible` predicate of `forward_chunk_scoped`), so it
/// always launches eager.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4ArLaunch {
    /// Eager HIP launches (retained backend off or poisoned, or the bundle is
    /// never a retained candidate).
    Eager,
    /// Recorded into, or replayed from, a retained body (or hipGraph-captured).
    Recorded,
}

/// The AR decode launch mode from the controller's actual state.
///
/// `retained_candidate` is the bundle-level part of `forward_chunk_scoped`'s
/// `eligible` predicate (no EP, Flash-Next profile, no lane scope, no QSA tap,
/// no QSA projection hook); the per-call parts (`n == 1`, no wide-hidden
/// capture, no invariant scope) are assumed to hold for the AR forward this
/// verify is compared against. Only [`ReplayState::Hip`] /
/// [`ReplayState::Fallback`] (the `hip_body` states) or `request == Hip` run AR
/// eager; every other state (`Armed`: the next eligible forward records;
/// `RecordingWarmup`, `Captured`, `ShadowValidated`, `Ready`) records or
/// replays. `Shadow` requests are classed `Recorded` conservatively (the
/// capture forward records; the route after validation is a developer
/// selection).
pub(crate) fn qwen4_ar_launch(
    retained_candidate: bool,
    request: rdna_compute::replay::ReplayBackendRequest,
    state: ReplayState,
    graph_capture: bool,
) -> Qwen4ArLaunch {
    use rdna_compute::replay::ReplayBackendRequest;
    if graph_capture {
        return Qwen4ArLaunch::Recorded;
    }
    if !retained_candidate
        || request == ReplayBackendRequest::Hip
        || matches!(state, ReplayState::Hip | ReplayState::Fallback)
    {
        return Qwen4ArLaunch::Eager;
    }
    Qwen4ArLaunch::Recorded
}

pub(crate) fn qwen4_spec_logit_rows(max_chunk: usize) -> usize {
    QWEN4_SPEC_VERIFY_ROWS.min(max_chunk)
}

/// Qualified numeric verify width: the Single Flash-Next scope keeps `rows`;
/// every other scope clamps to the proven `FEW_ROW_ROUTE_ROWS` bound. The CPU
/// window still narrows either case. Pure numeric cap only: the caller owns
/// the eligibility predicate.
fn qwen4_scoped_verify_rows(
    rows: usize,
    eligible_single_fn: bool,
    cpu_window_rows: Option<usize>,
) -> usize {
    let qualified = if eligible_single_fn {
        rows
    } else {
        rows.min(rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS)
    };
    cpu_window_rows.map_or(qualified, |cpu_rows| qualified.min(cpu_rows))
}

/// Allocation plan of one [`Qwen4GpuForwardScratch`].
///
/// `Full` is the historical plan: one dedicated allocation per field, every
/// MoE buffer sized for `max_chunk` rows of the widest (F32) route.
/// gfx942 deliberately uses `Full`: its FP8 MFMA route needs the full F32
/// scatter/combine buffers, not the gfx1201 symmetric IU4 storage aliases.
/// `FnSymGfx1201` is the proven-route plan of a gfx1201 Flash-Next artifact
/// whose routed headers are all symmetric: the MoE buffers are sized for the
/// symmetric IU4 prefill route plus a [`QWEN4_COMPACT_FALLBACK_ROWS`]-row F32
/// fallback, and the step-local fields of the GDN, QSA, HC, PLE, MoE and
/// embedding steps share one phase arena (see
/// [`Qwen4GpuForwardScratch::plan`]).  Both modes run identical
/// arithmetic; the mode only changes where intermediates live, so it must be
/// decided before expert placement and frozen for the forward's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ForwardScratchMode {
    Full,
    FnSymGfx1201,
}

/// Identical native trunk padding scope for sequential and paired prefill.
fn native_trunk_prefill_scope(
    prefill: bool,
    iu4: bool,
    arch: &str,
    mode: Qwen4ForwardScratchMode,
) -> bool {
    prefill && iu4 && arch == "gfx1151" && mode == Qwen4ForwardScratchMode::Full
}

/// gfx942 pins canonical prefill routing; gfx1151 retains its opt-in profile.
fn trunk_prefill_invariant_scope(prefill: bool, arch: &str, rdna_opt_in: bool) -> bool {
    prefill && (arch == "gfx942" || (arch == "gfx1151" && rdna_opt_in))
}

/// The canonical-routing scope flags `(pm_invariant, append)` of a forward of
/// `rows` rows over explicit knob values (`HIPFIRE_QWEN4_APPEND_WMMA`, its
/// `_MIN_ROWS`, `HIPFIRE_PREFILL_PM_INVARIANT`): the one pure decision
/// [`Qwen4GpuForward`]'s forward and the placement census of the lazy F16
/// weight shadows (`expert_residency::f16_shadow_route_rows`) both read, so
/// the rows a shadow-creating route sees cannot drift from the census.
pub(crate) fn qwen4_route_scope_flags(
    prefill_scope: bool,
    arch: &str,
    rows: usize,
    append_enabled: bool,
    append_min_rows: usize,
    pm_invariant_enabled: bool,
) -> (bool, bool) {
    let pm_invariant = trunk_prefill_invariant_scope(
        prefill_scope,
        arch,
        arch == "gfx1151" && pm_invariant_enabled,
    );
    let append = rdna_compute::prefill_invariant::qwen4_append_wmma_route(
        append_enabled,
        prefill_scope,
        arch,
        rows,
        append_min_rows,
        pm_invariant,
    );
    (pm_invariant, append)
}

/// Rows the `FnSymGfx1201` plan sizes the F32 fallback MoE buffers for.  A
/// larger MoE invocation that is not on the symmetric IU4 route has no buffer
/// to run in; dispatch refuses it before its first launch.
pub const QWEN4_COMPACT_FALLBACK_ROWS: usize = 512;

/// Storage-only knob of the compact scratch, read once as a tri-state:
/// unset leaves the route's default (on for W3/MQ4 expert-slab artifacts and,
/// with the IU4 route requested, for symmetric-IU4 packed ones), `1` forces it
/// on where the route is eligible, `0` keeps the full allocation plan (the IU4
/// math stays enabled either way).
pub const QWEN4_COMPACT_SCRATCH_ENV: &str = "HIPFIRE_QWEN4_COMPACT_SCRATCH";

/// Developer knob: make the compact-scratch acceptance check in
/// `Qwen4GpuForward::new_inner` fail as if one routed layer were unverified
/// (default off).  Per-layer verification and routes are unchanged, so the
/// Full fallback computes exactly what a Full-mode run computes.
pub const QWEN4_COMPACT_FORCE_UNVERIFIED_ENV: &str = "HIPFIRE_QWEN4_COMPACT_FORCE_UNVERIFIED";

/// Pure predicate of [`qwen4_forward_scratch_mode`]: every term must hold
/// for the compact plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompactScratchTerms {
    /// `None` unset, `Some(true)` `1`, `Some(false)` `0`.
    pub knob: Option<bool>,
    pub exact_gfx1201: bool,
    pub grouped_gemm: bool,
    pub sym_iu4_requested: bool,
    pub source_symmetric: bool,
    /// Every routed trunk layer is a slab arena (manifest recipe
    /// `FlashNextSlab`).
    pub source_slab: bool,
    pub ple_key_value_bf16: bool,
    pub flash_next_profile: bool,
}

impl CompactScratchTerms {
    /// Exactly gfx1201: no other arch, and no family prefix match.
    pub(crate) fn is_exact_gfx1201(arch: &str) -> bool {
        arch == "gfx1201"
    }

    /// The knob as a tri-state with `developer_bool`'s spellings: `1` is
    /// `Some(true)`, `0` is `Some(false)`, anything else (including unset)
    /// is `None`.
    pub(crate) fn knob_from_env() -> Option<bool> {
        if hipfire_config::developer_bool(QWEN4_COMPACT_SCRATCH_ENV, false) {
            Some(true)
        } else if !hipfire_config::developer_bool(QWEN4_COMPACT_SCRATCH_ENV, true) {
            Some(false)
        } else {
            None
        }
    }

    pub(crate) fn mode(self) -> Qwen4ForwardScratchMode {
        // Both routes default on (`0` opts out): expert-slab artifacts, and
        // symmetric-IU4 packed artifacts with the IU4 route requested (the
        // load assumes symmetry; the GPU check at attach confirms or re-plans
        // Full).
        let route = (self.source_slab || (self.source_symmetric && self.sym_iu4_requested))
            && self.knob != Some(false);
        if route
            && self.exact_gfx1201
            && self.grouped_gemm
            && self.ple_key_value_bf16
            && self.flash_next_profile
        {
            Qwen4ForwardScratchMode::FnSymGfx1201
        } else {
            Qwen4ForwardScratchMode::Full
        }
    }
}

/// The device, flag and profile terms of [`qwen4_forward_scratch_mode`]
/// with the artifact-dependent terms assumed true.  `false` means the mode is
/// `Full` whatever the artifact holds, so a caller can skip its payload
/// header scan.
pub fn qwen4_compact_scratch_possible(gpu: &Gpu, config: &Qwen4Config) -> bool {
    qwen4_forward_scratch_mode(gpu, config, true, true, true)
        == Qwen4ForwardScratchMode::FnSymGfx1201
}

/// The scratch plan a load takes: `FnSymGfx1201` for an exact gfx1201
/// Flash-Next artifact whose MoE layers all run a route the compact buffers
/// were sized for.  `source_slab` is the manifest's `FlashNextSlab` recipe
/// (every routed trunk layer is a slab arena; default on, `0` opts out);
/// `source_symmetric` is the proof from the artifact's actual routed payload
/// headers ([`crate::artifact::routed_headers_symmetric`]), not metadata, and
/// also needs the IU4 route requested (default on, `0` opts out);
/// `ple_key_value_bf16` states that the PLE key/value projections are BF16.
/// Cheap arch/flag checks come first so a caller may skip its header scan
/// when this would be `Full` regardless.  [`Qwen4GpuForward::new`] still
/// confirms the route on the loaded layers and refuses a `FnSymGfx1201`
/// forward the GPU check does not fully confirm.
pub fn qwen4_forward_scratch_mode(
    gpu: &Gpu,
    config: &Qwen4Config,
    source_symmetric: bool,
    source_slab: bool,
    ple_key_value_bf16: bool,
) -> Qwen4ForwardScratchMode {
    // Freeze the gfx942 storage rung independently of row count or IU4 knobs.
    if gpu.arch == "gfx942" {
        return Qwen4ForwardScratchMode::Full;
    }
    static KNOB: std::sync::LazyLock<Option<bool>> =
        std::sync::LazyLock::new(CompactScratchTerms::knob_from_env);
    CompactScratchTerms {
        knob: *KNOB,
        exact_gfx1201: CompactScratchTerms::is_exact_gfx1201(&gpu.arch),
        grouped_gemm: gpu.flags.moe_grouped_gemm,
        sym_iu4_requested: gpu.qwen4_moe_sym_iu4_requested(),
        source_symmetric,
        source_slab,
        ple_key_value_bf16,
        flash_next_profile: matches!(config.profile, crate::config::Qwen4Profile::FlashNext),
    }
    .mode()
}

/// Device bytes of the forward resources that scale with the chunk: the
/// reusable scratch for `rows` rows plus the speculative logits and argmax.
/// Load admission and the auto expert placement reserve size against this.
pub fn qwen4_forward_device_bytes(
    config: &Qwen4Config,
    rows: usize,
    mode: Qwen4ForwardScratchMode,
) -> Option<u64> {
    let scratch = Qwen4GpuForwardScratch::device_bytes(config, rows, mode).ok()?;
    let [logits, argmax] = qwen4_spec_buffer_bytes(config, rows)?;
    scratch.checked_add(logits)?.checked_add(argmax)
}

/// Bytes of the two speculative buffers `Qwen4Bundle::attach_forward`
/// allocates beside the scratch for a `rows`-row forward: the F32 logits
/// (`spec_rows * vocab * 4`) and the i32 argmax (`spec_rows * 4`), each one
/// `gpu.zeros` allocation. `None` on arithmetic overflow.
fn qwen4_spec_buffer_bytes(config: &Qwen4Config, rows: usize) -> Option<[u64; 2]> {
    let spec_rows = qwen4_spec_logit_rows(rows) as u64;
    let logits = spec_rows.checked_mul((config.vocab_size as u64).checked_mul(4)?)?;
    let argmax = spec_rows.checked_mul(std::mem::size_of::<i32>() as u64)?;
    Some([logits, argmax])
}

/// Every device allocation [`qwen4_forward_device_bytes`] counts, one entry
/// per actual allocation: the owning scratch slots of the whole-model plan
/// (borrowed alias views and zero-byte slots are not allocations), then the
/// speculative logits and argmax buffers. The entries sum to
/// [`qwen4_forward_device_bytes`]; `None` on any overflow or a refused plan.
/// CPU-only: plans, never touches a device.
pub(crate) fn qwen4_forward_allocation_sizes(
    config: &Qwen4Config,
    rows: usize,
    mode: Qwen4ForwardScratchMode,
) -> Option<Vec<u64>> {
    let (slots, _) = Qwen4GpuForwardScratch::plan(config, rows, mode).ok()?;
    let [logits, argmax] = qwen4_spec_buffer_bytes(config, rows)?;
    let mut sizes = Vec::with_capacity(slots.len() + 2);
    for bytes in Qwen4GpuForwardScratch::owning_slot_bytes(&slots) {
        sizes.push(bytes.ok()?);
    }
    for bytes in [logits, argmax] {
        if bytes > 0 {
            sizes.push(bytes);
        }
    }
    Some(sizes)
}

/// Rows of the dedicated (resident) scratch bank of a decode-share load: decode, verify,
/// lanes up to this total, and every prefill tile below it run there (verdict change 2).
pub const QWEN4_DECODE_SHARE_RESIDENT_ROWS: usize = 512;

/// Alignment of every root start inside a borrowed donor span.
const DECODE_SHARE_ROOT_ALIGN: u64 = 256;

/// Where one owning scratch root of a borrowed prefill bank lives inside the donor span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchRootPlacement {
    pub field: ScratchField,
    pub offset: u64,
    pub bytes: u64,
}

/// The donor-span layout of one borrowed prefill bank ([`Qwen4GpuForwardScratch::borrowed_backing`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScratchBacking {
    pub capacity_rows: usize,
    /// [`ScratchField`] order, every owning non-empty slot EXCEPT `TokenIds`.
    pub roots: Vec<ScratchRootPlacement>,
    /// End of the last root rounded up to 256.
    pub span_bytes: u64,
}

/// Capacity classes of the borrowed prefill bank for a load whose prompt chunk is `max_chunk`:
/// powers of two from 1024 up to (excluding) `max_chunk`, then `max_chunk` itself; each class must
/// be > [`QWEN4_DECODE_SHARE_RESIDENT_ROWS`]. A borrow epoch whose largest tile has `r` rows uses the
/// smallest class `>= r`, so a 1,000-row prompt dirties (and restores) about 1/16 of the tail.
pub fn qwen4_decode_share_classes(max_chunk: usize) -> Vec<usize> {
    let mut classes = Vec::new();
    let mut class = 1024usize;
    while class < max_chunk {
        classes.push(class);
        match class.checked_mul(2) {
            Some(next) => class = next,
            None => break,
        }
    }
    if max_chunk > QWEN4_DECODE_SHARE_RESIDENT_ROWS {
        classes.push(max_chunk);
    }
    classes
}

/// Device bytes a decode-share load charges for its forward instead of
/// [`qwen4_forward_device_bytes`]`(config, max_chunk, mode)`: the resident bank
/// (`device_bytes(config, QWEN4_DECODE_SHARE_RESIDENT_ROWS, mode)`), plus a persistent
/// `TokenIds` owner for `max_chunk` rows if the resident plan's `TokenIds` is smaller (charged
/// whole, so the sum never undercounts), plus the speculative buffers of `max_chunk` rows.
pub fn qwen4_decode_share_dedicated_bytes(
    config: &Qwen4Config,
    max_chunk: usize,
    mode: Qwen4ForwardScratchMode,
) -> Option<u64> {
    let resident =
        Qwen4GpuForwardScratch::device_bytes(config, QWEN4_DECODE_SHARE_RESIDENT_ROWS, mode)
            .ok()?;
    let token_ids = |rows: usize| -> Option<u64> {
        let (slots, _) = Qwen4GpuForwardScratch::plan(config, rows, mode).ok()?;
        u64::try_from(slots[ScratchField::TokenIds as usize].bytes().ok()?).ok()
    };
    let persistent_ids = {
        let chunk = token_ids(max_chunk)?;
        if token_ids(QWEN4_DECODE_SHARE_RESIDENT_ROWS)? < chunk {
            chunk
        } else {
            0
        }
    };
    let [logits, argmax] = qwen4_spec_buffer_bytes(config, max_chunk)?;
    resident
        .checked_add(persistent_ids)?
        .checked_add(logits)?
        .checked_add(argmax)
}

/// Bytes of the donor span: `borrowed_backing(config, max_chunk, ..).span_bytes`.
pub fn qwen4_decode_share_borrow_bytes(
    config: &Qwen4Config,
    max_chunk: usize,
    mode: Qwen4ForwardScratchMode,
) -> Option<u64> {
    Qwen4GpuForwardScratch::borrowed_backing(
        config,
        max_chunk,
        mode,
        &Qwen4BandScope::single(config),
    )
    .ok()
    .map(|backing| backing.span_bytes)
}

/// The donor-tail byte prefix each prefill class dirties, in class order
/// (ascending): `shadow_bytes` (the borrowed weight-shadow region at the START
/// of the donor tail, [`crate::decode_share_shadows`]; 0 when the shadows stay
/// resident) plus [`qwen4_decode_share_borrow_bytes`] of every
/// [`qwen4_decode_share_classes`] entry for the chunk (the scratch backing that
/// follows the region). The single authority: the loader sizes the donor tail
/// (`S` of `with_decode_share`) and plans the BorrowPrefill class views from
/// these, the forward checks its banks and the loan's dirty extents against them.
pub fn qwen4_decode_share_class_spans(
    config: &Qwen4Config,
    max_chunk: usize,
    mode: Qwen4ForwardScratchMode,
    shadow_bytes: u64,
) -> Option<Vec<u64>> {
    qwen4_decode_share_classes(max_chunk)
        .into_iter()
        .map(|rows| qwen4_decode_share_borrow_bytes(config, rows, mode)?.checked_add(shadow_bytes))
        .collect()
}

/// Free device memory a chunk rung must leave beside its chunk-sized
/// resources for the kernels' lazily sized workspaces (333-347 MiB at
/// pp8192 on gfx1201, measured as load-time free minus prefill-time free).
pub const QWEN4_FORWARD_HEADROOM_BYTES: u64 = 1 << 30;

const QWEN4_STEP_INLINE_CAPACITY: usize = 384;
const QWEN4_QSA_INLINE_CAPACITY: usize = 12;

/// Long-prefill host-expert staging (G2). `1`/`0` force it on/off; unset it
/// is on for gfx1201 (the card it was measured on, whatever its link) and
/// for the borrowed-donor stage (packed aggregate and slab-arena layouts) on
/// any discrete device whose measured H2D link is narrow, below 16 GB/s
/// ([`crate::host_link::expert_stage_default`]); it stays off for unified
/// memory and when the link could not be measured.
/// The stage switch itself never changes placement: the expert placement
/// charges the probe-free switch
/// ([`crate::expert_residency::expert_stage_placement_enabled`], explicit
/// value else gfx1201), not this link-aware default. A
/// staged prefill borrows the VRAM of resident expert layers as its two DMA
/// stages (no dedicated VRAM: the last two resident layers of a packed
/// artifact, the latest resident slab layer of each parity that holds a cold
/// arena for a slab artifact, or the parity's largest resident slab layer
/// when it is within a sixteenth of its largest cold arena, whose overflowing
/// cold layers then read their trailing experts in place; a tier layout,
/// which would allocate dedicated buffers, stages only under the probe-free
/// switch), and restores them from
/// pinned host copies (additional host RAM, separate from the placement's
/// pinned experts) before the forward returns, so decode keeps today's
/// residency. Under the probe-free switch a slab artifact keeps its charged
/// dedicated stage buffers instead.
pub const QWEN4_EXPERT_STAGE_ENV: &str = "HIPFIRE_QWEN4_EXPERT_STAGE";

/// Forwards of at least this many rows stage ([`QWEN4_EXPERT_STAGE_ENV`]);
/// smaller ones, decode and verify keep the mapped reads.
pub const QWEN4_EXPERT_STAGE_MIN_ROWS_ENV: &str = "HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS";

/// Whether a long forward that captures wide hidden rows (the native-MTP
/// prompt fill's trunk tiles) stages like any other long forward. `0` keeps
/// those tiles on the mapped reads; unset or `1` stages them. The capture
/// only copies the trunk's output streams after the program has run, on the
/// compute stream, and never reads an expert, so it is byte-identical either
/// way. Verify forwards capture too but stay below the row threshold.
pub const QWEN4_EXPERT_STAGE_CAPTURE_ENV: &str = "HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE";

/// One Flash-Next routed expert's QT44 gate/up and QT53 down bytes.
const QWEN4_EXPERT_STAGE_GATE_UP_BYTES: usize = 1_740_800;
const QWEN4_EXPERT_STAGE_DOWN_BYTES: usize = 870_400;

/// Where the effective stage `min_rows` came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StageMinRowsSource {
    Env,
    Default,
}

impl StageMinRowsSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Default => "default",
        }
    }
}

/// Effective stage threshold and its source. `chunk_rows` is the forward's
/// prefill chunk capacity (the largest forward it will ever see). An explicit
/// [`QWEN4_EXPERT_STAGE_MIN_ROWS_ENV`] (integer >= 2) is authoritative. The
/// default is `min(2048, chunk_rows)`, so a full chunk always qualifies on an
/// arch whose chunk is below 2048 (gfx1100 on a fast link: 1536); on a
/// `narrow_link` (measured narrow discrete host link) off gfx1201 it is a
/// fixed 768, never lowered to the chunk, so forwards under 768 rows (and any
/// chunk admitted below 768) keep the mapped reads. gfx1201 keeps 2048
/// (K3: 1536-row forwards expose the DMA) and whatever its link, so its
/// behavior is unchanged.
fn resolve_expert_stage_min_rows(
    arch: &str,
    chunk_rows: usize,
    narrow_link: bool,
    env: Option<&str>,
) -> (usize, StageMinRowsSource) {
    if let Some(rows) = env
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&rows| rows >= 2)
    {
        return (rows, StageMinRowsSource::Env);
    }
    let rows =
        rdna_compute::expert_capabilities::stage_default_min_rows(arch, chunk_rows, narrow_link);
    (rows, StageMinRowsSource::Default)
}

fn qwen4_expert_stage_min_rows(gpu: &Gpu, chunk_rows: usize) -> (usize, StageMinRowsSource) {
    expert_stage_min_rows_from(
        &gpu.arch,
        chunk_rows,
        hipfire_config::developer_var(QWEN4_EXPERT_STAGE_MIN_ROWS_ENV)
            .ok()
            .as_deref(),
        || crate::host_link::host_link(gpu),
    )
}

/// [`qwen4_expert_stage_min_rows`] with the `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS`
/// value and the (lazily measured) host link supplied. gfx1201 and an explicit
/// threshold never evaluate `link`, so they never probe.
pub(crate) fn expert_stage_min_rows_from(
    arch: &str,
    chunk_rows: usize,
    env: Option<&str>,
    link: impl FnOnce() -> crate::host_link::HostLink,
) -> (usize, StageMinRowsSource) {
    let narrow_link = rdna_compute::expert_capabilities::stage_reads_link(arch)
        && !env
            .and_then(|value| value.trim().parse::<usize>().ok())
            .is_some_and(|rows| rows >= 2)
        && link().is_narrow_discrete();
    resolve_expert_stage_min_rows(arch, chunk_rows, narrow_link, env)
}

/// Whether long-prefill expert staging would stage a window of at most
/// [`hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS`] rows: staging on and
/// `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` forced to the cap or below. Every
/// default threshold (2048 on gfx1201, the fixed 768 of a narrow link,
/// `min(2048, prefill chunk)` otherwise) is hundreds of rows at least, so only
/// the forced value can reach a narrow window and the link is irrelevant here
/// (resolved as not narrow, which needs no probe). A staged forward
/// swaps the layer's pointer tables for the stage's, which the CPU tier split's
/// masked route does not cover, so the split refuses to arm (one log line)
/// instead of failing each narrow call.
fn expert_stage_takes_narrow_windows(
    arch: &str,
    stage_on: bool,
    min_rows_env: Option<&str>,
) -> bool {
    stage_on
        && resolve_expert_stage_min_rows(arch, usize::MAX, false, min_rows_env).0
            <= hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS
}

/// Which rows of a batched Qwen4 prefill write language-model logits.
///
/// `All` is the public chunk/capture contract.  `Final` keeps the complete
/// trunk batch but emits only its final row for ordinary autoregressive AR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4OutputRows {
    All,
    Final,
}

impl Qwen4OutputRows {
    pub(crate) fn count(self, n: usize) -> usize {
        match self {
            Self::All => n,
            Self::Final => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qwen4OutputPolicy {
    Rows(Qwen4OutputRows),
    None,
}

impl Qwen4OutputPolicy {
    fn requested_rows(self, n: usize) -> Option<usize> {
        match self {
            Self::Rows(rows) => Some(rows.count(n)),
            Self::None => None,
        }
    }
}

/// The head a mesh output band writes after its trunk: the logits (and optional
/// per-row argmax) destination for `rows` of the chunk. `None` at the call
/// discards (non-final prefill chunks); a non-output band never has one.
#[derive(Clone, Copy)]
pub struct Qwen4MeshHead<'a> {
    pub logits: &'a GpuTensor,
    pub top1: Option<&'a GpuTensor>,
    pub rows: Qwen4OutputRows,
}

/// Exact geometry of the HC streams a band publishes/receives for one mesh
/// epoch: the routing rows the epoch's Gpu scope selects, the stream dtype
/// (`bf16_streams`) and the live bytes (`rows * hc_count * hidden * dtype`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4MeshLayout {
    pub route_rows: usize,
    pub bf16_streams: bool,
    pub stream_bytes: usize,
}

/// The steps one mesh call runs of its band program (`mesh_op_range`). A TP
/// rank stops at every collective, so its program runs in segments: only the
/// `first` uploads and embeds the tokens and stages the PLE rows, only the
/// `last` runs the head and the wide capture and commits the state. A PP band
/// runs its whole program as one segment.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MeshSegment {
    steps: std::ops::Range<usize>,
    first: bool,
    last: bool,
}

/// Host-side forward phases timed when profiling is enabled.
#[derive(Clone, Copy)]
pub(crate) enum Qwen4ProfilePhase {
    MoeSeal,
    PleWait,
    PleStage,
    PleUpload,
    PleApply,
}

impl Qwen4ProfilePhase {
    pub(crate) const ALL: [Self; 5] = [
        Self::MoeSeal,
        Self::PleWait,
        Self::PleStage,
        Self::PleUpload,
        Self::PleApply,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::MoeSeal => "moe_seal",
            Self::PleWait => "ple_wait",
            Self::PleStage => "ple_stage",
            Self::PleUpload => "ple_upload",
            Self::PleApply => "ple_apply",
        }
    }
}

/// Calls and host nanoseconds per [`Qwen4ProfilePhase`], indexed by phase.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Qwen4ProfileStats {
    pub(crate) calls: [u64; Qwen4ProfilePhase::ALL.len()],
    pub(crate) ns: [u64; Qwen4ProfilePhase::ALL.len()],
}

thread_local! {
    static QWEN4_PROFILE_ENABLED: Cell<bool> = const { Cell::new(false) };
    static QWEN4_PROFILE_STATS: RefCell<Qwen4ProfileStats> =
        RefCell::new(Qwen4ProfileStats::default());
}

#[inline(always)]
fn qwen4_profile_enabled() -> bool {
    QWEN4_PROFILE_ENABLED.with(|enabled| enabled.get())
}

pub(crate) fn qwen4_profile_enable(enabled: bool) {
    QWEN4_PROFILE_ENABLED.with(|current| current.set(enabled));
}

pub(crate) fn qwen4_profile_reset() {
    QWEN4_PROFILE_STATS.with(|stats| *stats.borrow_mut() = Qwen4ProfileStats::default());
}

pub(crate) fn qwen4_profile_snapshot() -> Qwen4ProfileStats {
    QWEN4_PROFILE_STATS.with(|stats| *stats.borrow())
}

#[inline(always)]
fn qwen4_profile_start() -> Option<Instant> {
    if qwen4_profile_enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline(always)]
fn qwen4_profile_record(phase: Qwen4ProfilePhase, started: Option<Instant>) {
    let Some(started) = started else {
        return;
    };
    let elapsed = started.elapsed().as_nanos() as u64;
    QWEN4_PROFILE_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        let slot = phase as usize;
        stats.calls[slot] = stats.calls[slot].saturating_add(1);
        stats.ns[slot] = stats.ns[slot].saturating_add(elapsed);
    });
}

/// Errors returned by the native ordinary-HIP path.
#[derive(Debug)]
pub enum Qwen4GpuForwardError {
    Hip(hip_bridge::HipError),
    Weights(WeightError),
    Invalid(String),
    /// The compact scratch plan was selected before the payload was proven
    /// symmetric and the GPU check verified fewer routed layers than it needs.
    /// Everything `new_inner` allocated was released before this was returned.
    CompactUnverified {
        verified: usize,
        total: usize,
    },
    Dispatch(String),
    Ple(String),
}

impl fmt::Display for Qwen4GpuForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => write!(f, "Qwen4 GPU forward HIP error: {error}"),
            Self::Weights(error) => write!(f, "Qwen4 GPU forward weights error: {error}"),
            Self::Invalid(error) => write!(f, "Qwen4 GPU forward invalid input: {error}"),
            Self::CompactUnverified { verified, total } => write!(
                f,
                "Qwen4 GPU forward: compact scratch was selected, but the GPU symmetry check \
                 verified only {verified}/{total} layers symmetric"
            ),
            Self::Dispatch(error) => write!(f, "Qwen4 GPU forward dispatch error: {error}"),
            Self::Ple(error) => write!(f, "Qwen4 GPU forward PLE error: {error}"),
        }
    }
}

impl std::error::Error for Qwen4GpuForwardError {}

impl From<hip_bridge::HipError> for Qwen4GpuForwardError {
    fn from(error: hip_bridge::HipError) -> Self {
        Self::Hip(error)
    }
}

impl From<WeightError> for Qwen4GpuForwardError {
    fn from(error: WeightError) -> Self {
        Self::Weights(error)
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> Qwen4GpuForwardError {
    Qwen4GpuForwardError::Invalid(message.into())
}

fn checked_bytes(rows: usize, stride: usize) -> Result<usize, Qwen4GpuForwardError> {
    crate::projection::checked_bytes(rows, stride)
        .ok_or_else(|| invalid("packed row byte size overflow"))
}

pub(crate) fn f32_view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    debug_assert_eq!(tensor.dtype, DType::F32);
    tensor.sub_offset(offset, len)
}

pub(crate) fn view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    tensor.sub_offset(offset, len)
}

/// Byte range `(offset, len)` in `scratch.token_ids` of `tokens` at row
/// `offset`, when the forward's upload (`uploaded_rows` rows mirrored in
/// `host_bytes`, native-endian i32) holds exactly those ids there.
fn uploaded_ids_range(
    uploaded_rows: usize,
    host_bytes: &[u8],
    offset: usize,
    tokens: &[u32],
) -> Option<(usize, usize)> {
    const ID_BYTES: usize = std::mem::size_of::<i32>();
    let end = offset.checked_add(tokens.len())?;
    if tokens.is_empty() || end > uploaded_rows {
        return None;
    }
    let start_byte = offset.checked_mul(ID_BYTES)?;
    let end_byte = end.checked_mul(ID_BYTES)?;
    let mirror = host_bytes.get(start_byte..end_byte)?;
    mirror
        .chunks_exact(ID_BYTES)
        .zip(tokens)
        .all(|(bytes, &token)| bytes == (token as i32).to_ne_bytes())
        .then_some((start_byte, end_byte - start_byte))
}

pub(crate) fn matrix_view(
    tensor: &GpuTensor,
    rows: usize,
    columns: usize,
) -> Result<GpuTensor, Qwen4GpuForwardError> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("Qwen4 matrix view shape overflows"))?;
    let mut view = tensor.sub_offset(0, elements);
    view.shape = vec![rows, columns];
    Ok(view)
}

fn weight_dims(reference: &TensorRef) -> Result<(usize, usize), Qwen4GpuForwardError> {
    if reference.shape.len() != 2 {
        return Err(invalid(format!(
            "projection {} has logical shape {:?}, expected rank two",
            reference.name, reference.shape
        )));
    }
    Ok((reference.shape[0], reference.shape[1]))
}
pub(crate) fn dense_ref<'a>(
    weights: &'a Qwen4Weights,
    reference: &TensorRef,
) -> Result<WeightRef<'a>, Qwen4GpuForwardError> {
    let tensor = weights.resident(reference)?;
    let (m, k) = weight_dims(reference)?;
    Ok(WeightRef {
        buf: tensor,
        dtype: tensor.dtype,
        m,
        k,
        row_stride: row_stride(tensor.dtype, k),
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    })
}

/// The admitted Qwen4 grouped route is declared by this architecture
/// constructor and consumed as data by shared sealing/lowering.  The shared
/// expert's formats are those of the bound weights: BF16, or the Q8_0 decode
/// copies (`Qwen4GpuForward::decode_q8`). `capability` is the layer's
/// declaration ([`Qwen4MoeLayerRuntime::route_capability`]): the QT44/QT53
/// grouped route, its symmetric variant (a prefill chunk over a layer whose
/// expert headers were verified symmetric, which the opt-in IU4 arm
/// requires), or the gfx942 qt=55 [`MoeRouteCapability::Fp8B128Grouped`]
/// route, whose routed gate_up and down formats are both `FP8E4M3B128`.
///
/// `slab_cold_codec` is a sealed expert-slab layer's cold codec id (W3 = 1,
/// E8P = 3), read from the layer's load-time slab metadata
/// ([`Qwen4MoeLayerRuntime::slab_cold_codec`], no table scan); `None` for an
/// MQ4-only slab layer and for every non-slab route.
fn qwen4_route_policy(
    config: &Qwen4Config,
    shared: &MoeSharedWeights<'_>,
    capability: MoeRouteCapability,
    slab_cold_codec: Option<u8>,
    expert_keep_mask: usize,
) -> MoeRoutePolicy {
    let (routed_gate_up, routed_down) = if capability.is_fp8_grouped() {
        (DType::FP8E4M3B128, DType::FP8E4M3B128)
    } else {
        (DType::MQ4G256V2, DType::MQ4G128V2)
    };
    MoeRoutePolicy {
        expert_keep_mask,
        capability,
        geometry: MoeRouteGeometry {
            experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hidden: config.hidden_size,
            routed_intermediate: config.moe_intermediate_size,
            shared_intermediate: config.shared_expert_intermediate_size,
        },
        formats: MoeRouteFormats {
            router: DType::BF16,
            shared_selector: DType::BF16,
            shared_gate: shared.gate.dtype,
            shared_up: shared.up.dtype,
            shared_down: shared.down.dtype,
            routed_gate_up,
            routed_down,
        },
        slab_cold_codec,
    }
}
pub(crate) fn program_dims(config: &Qwen4Config) -> Qwen4ProgramDims {
    Qwen4ProgramDims {
        hidden: config.hidden_size,
        hc_count: config.hc_count,
        hc_lowrank: config.hc_lowrank,
        indexer_n_heads: config.indexer_n_heads,
        indexer_kv_heads: config.indexer_kv_heads,
        indexer_head_dim: config.indexer_head_dim,
        indexer_budget: config.indexer_budget,
        indexer_compress_ratio: config.indexer_compress_ratio,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        linear_num_key_heads: config.linear_num_key_heads,
        linear_num_value_heads: config.linear_num_value_heads,
        linear_key_head_dim: config.linear_key_head_dim,
        linear_value_head_dim: config.linear_value_head_dim,
        linear_conv_kernel_dim: config.linear_conv_kernel_dim,
        ffn: match &config.ffn {
            Qwen4FfnConfig::Dense { intermediate_size } => Qwen4FfnDims::Dense {
                intermediate: *intermediate_size,
            },
            Qwen4FfnConfig::Moe {
                num_experts,
                top_k,
                intermediate_size,
                shared_intermediate_size,
                ..
            } => Qwen4FfnDims::Moe {
                num_experts: *num_experts,
                experts_per_token: *top_k,
                intermediate: *intermediate_size,
                // A shared-expert-less MoE has no executor; `validate_dims`
                // rejects the zero extent.
                shared_intermediate: shared_intermediate_size.unwrap_or(0),
            },
        },
        ple_conv_kernel_dim: config.ple_conv_kernel_size,
        norm_eps: EPSILON,
    }
}

/// Columns of the router-logit scratch view (`num_experts`; one placeholder
/// column for a dense FFN, which routes nothing).
pub(crate) fn router_columns(dims: Qwen4ProgramDims) -> usize {
    match dims.ffn {
        Qwen4FfnDims::Moe { num_experts, .. } => num_experts,
        Qwen4FfnDims::Dense { .. } => 1,
    }
}

/// Q8_0 decode copies per layer (`Qwen4GpuForward::decode_q8`).
pub(crate) const DECODE_Q8_PER_LAYER: usize = 7;

/// A decode-shaped forward's projection (one token, or the few rows of a
/// speculative verify, which must match it bitwise): its Q8_0 decode copy
/// (`Qwen4GpuForward::decode_q8`) when one exists. `rows` is the routing row
/// count (`Gpu::route_rows`) and `few_row_limit` the architecture's admitted
/// exact cap (`Gpu::few_row_limit`): the existing copy binds exactly while
/// `rows <= few_row_limit`; nothing is allocated or requantized per call.
pub(crate) fn decode_copy<'a>(
    copies: &'a [GpuTensor],
    index: usize,
    rows: usize,
    few_row_limit: usize,
    source: WeightRef<'a>,
) -> WeightRef<'a> {
    match copies.get(index) {
        Some(buf) if rows <= few_row_limit => WeightRef {
            buf,
            dtype: DType::Q8_0,
            row_stride: row_stride(DType::Q8_0, source.k),
            ..source
        },
        _ => source,
    }
}

pub(crate) fn hyper_read_desc<'a>(
    weights: &'a Qwen4Weights,
    hyper: &HyperConnectionReadWeights,
) -> Result<Qwen4HyperReadWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4HyperReadWeights {
        norm: weights.resident(&hyper.hc_norm)?,
        input_mix_down: dense_ref(weights, &hyper.input_mix_down)?,
        input_mix_up: dense_ref(weights, &hyper.input_mix_up)?,
    })
}

pub(crate) fn hyper_desc<'a>(
    weights: &'a Qwen4Weights,
    hyper: &HyperConnectionWeights,
) -> Result<Qwen4HyperWeights<'a>, Qwen4GpuForwardError> {
    let norm = weights.resident(&hyper.hc_norm)?;
    Ok(Qwen4HyperWeights {
        read: Qwen4HyperReadWeights {
            norm,
            input_mix_down: dense_ref(weights, &hyper.input_mix_down)?,
            input_mix_up: dense_ref(weights, &hyper.input_mix_up)?,
        },
        write: crate::program::Qwen4HyperWriteWeights {
            norm,
            block_inject: dense_ref(weights, &hyper.block_inject)?,
        },
    })
}

fn gdn_desc<'a>(
    weights: &'a Qwen4Weights,
    gdn: &crate::weights::GdnWeights,
) -> Result<Qwen4GdnWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4GdnWeights {
        qkv: dense_ref(weights, &gdn.qkv)?,
        conv: weights.resident(&gdn.conv)?,
        in_proj_a: dense_ref(weights, &gdn.in_proj_a)?,
        in_proj_b: dense_ref(weights, &gdn.in_proj_b)?,
        a_log: weights.resident(&gdn.a_log)?,
        dt_bias: weights.resident(&gdn.dt_bias)?,
        z: dense_ref(weights, &gdn.z)?,
        norm: weights.resident(&gdn.norm)?,
        output: dense_ref(weights, &gdn.output)?,
    })
}

pub(crate) fn qsa_desc<'a>(
    weights: &'a Qwen4Weights,
    qsa: &crate::weights::QsaWeights,
) -> Result<Qwen4QsaWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4QsaWeights {
        indexer_qk: dense_ref(weights, &qsa.indexer_qk)?,
        indexer_q_norm: weights.resident(&qsa.indexer_q_norm)?,
        indexer_k_norm: weights.resident(&qsa.indexer_k_norm)?,
        q: dense_ref(weights, &qsa.q)?,
        k: dense_ref(weights, &qsa.k)?,
        v: dense_ref(weights, &qsa.v)?,
        q_norm: weights.resident(&qsa.q_norm)?,
        k_norm: weights.resident(&qsa.k_norm)?,
        output: dense_ref(weights, &qsa.output)?,
    })
}
pub(crate) fn ple_desc<'a>(
    weights: &'a Qwen4Weights,
    ple: &crate::weights::PleWeights,
) -> Result<Qwen4PleWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4PleWeights {
        key: dense_ref(weights, &ple.key)?,
        value: dense_ref(weights, &ple.value)?,
        norm_key: weights.resident(&ple.norm_key)?,
        norm_query: weights.resident(&ple.norm_query)?,
        norm_conv: weights.resident(&ple.norm_conv)?,
        conv: weights.resident(&ple.conv)?,
    })
}

/// `HIPFIRE_QWEN4_PLE_FUSE`: whether a `rows`-row forward stages the PLE rows
/// straight as F16 for BF16 key/value projections. Admitted wide exact verify
/// keeps the singleton F32 input representation before staging, not merely
/// at projection dispatch; ordinary/invariant prefill keeps its F16 route.
pub(crate) fn qwen4_ple_rows_f16(
    gpu: &Gpu,
    bf16_state: bool,
    hc_count: usize,
    hidden_size: usize,
    ple: &Qwen4PleWeights<'_>,
    rows: usize,
) -> bool {
    let route_rows = gpu.route_rows(rows);
    if route_rows > rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS
        && route_rows <= gpu.few_row_limit()
    {
        return false;
    }
    bf16_state
        && gpu.flags.qwen4_ple_fuse_enabled()
        && hc_count <= 4
        && hidden_size % 32 == 0
        && ple.key.k == hidden_size
        && ple.value.k == hidden_size
        && gpu.qwen4_f16_wmma_applies(ple.key.buf, ple.key.k, rows)
        && gpu.qwen4_f16_wmma_applies(ple.value.buf, ple.value.k, rows)
}

/// `decode_q8` / `rows`: the forward's Q8_0 decode copies and row count; a
/// forward of at most `few_row_limit` rows (`Gpu::few_row_limit`) binds this
/// layer's shared-expert copies.  `moe` is the
/// layer's MoE runtime and is `Some` exactly for a MoE FFN.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_desc<'a>(
    weights: &'a Qwen4Weights,
    layer: &Qwen4LayerWeights,
    moe: Option<&'a Qwen4MoeLayerRuntime>,
    config: &Qwen4Config,
    decode_q8: &'a [GpuTensor],
    rows: usize,
    few_row_limit: usize,
    stage: Option<&'a Qwen4ExpertStage>,
) -> Result<Qwen4LayerDescription<'a>, Qwen4GpuForwardError> {
    let attention = match layer.kind {
        LayerType::LinearAttention => Qwen4AttentionWeights::Linear(gdn_desc(
            weights,
            layer
                .gdn
                .as_ref()
                .ok_or_else(|| invalid("GDN weights missing"))?,
        )?),
        LayerType::FullAttention => Qwen4AttentionWeights::Full(qsa_desc(
            weights,
            layer
                .attention
                .as_ref()
                .ok_or_else(|| invalid("QSA weights missing"))?,
        )?),
    };
    let ffn = match (&layer.ffn, moe) {
        (Qwen4FfnWeights::Dense(dense), None) => {
            let gate = dense_ref(weights, &dense.gate)?;
            Qwen4FfnBinding::Dense(Qwen4DenseBinding {
                intermediate: gate.m,
                gate,
                up: dense_ref(weights, &dense.up)?,
                down: dense_ref(weights, &dense.down)?,
            })
        }
        (Qwen4FfnWeights::Moe(_), Some(moe)) => {
            let copy = |slot: usize, source: &'a ProjectionView| {
                decode_copy(
                    decode_q8,
                    layer.layer * DECODE_Q8_PER_LAYER + slot,
                    rows,
                    few_row_limit,
                    source.dispatch_ref(),
                )
            };
            let shared = MoeSharedWeights {
                selector: moe.shared_scalar.dispatch_ref(),
                gate: copy(4, &moe.shared_gate),
                up: copy(5, &moe.shared_up),
                down: copy(6, &moe.shared_down),
            };
            Qwen4FfnBinding::Moe(Qwen4MoeBinding {
                cpu_deferred_experts: moe.cpu_deferred_experts,
                route_policy: qwen4_route_policy(
                    config,
                    &shared,
                    moe.route_capability(rows),
                    moe.slab_cold_codec(),
                    moe.expert_keep_mask
                        .as_ref()
                        .map_or(0, |mask| mask.buf.as_ptr() as usize),
                ),
                table: &moe.table,
                cache: &moe.cache,
                routed_experts: moe,
                router: moe.router.dispatch_ref(),
                shared,
                intermediate: moe
                    .experts
                    .first()
                    .map(|expert| expert.gate_up.m)
                    .unwrap_or_default()
                    / 2,
                experts_all_gate_up_mq4: moe.experts_all_gate_up_mq4,
                expert_gate_up_dtypes: moe
                    .slab
                    .as_ref()
                    .filter(|slab| slab.mixed())
                    .map(|slab| slab.gate_up_dtypes.as_slice()),
                expert_down_dtypes: moe
                    .slab
                    .as_ref()
                    .filter(|slab| slab.mixed())
                    .map(|slab| slab.down_dtypes.as_slice()),
                expert_codec_tags: moe.slab.as_ref().and_then(|slab| slab.codec_tags.as_ref()),
                expert_gate_up_ptrs: &moe.expert_gate_up_ptrs,
                expert_down_ptrs: &moe.expert_down_ptrs,
                expert_stage_ptrs: stage
                    .filter(|_| moe.host_mapped())
                    .map(|stage| stage.tables(layer.layer)),
                expert_gate_up_entries: &moe.expert_gate_up_entries,
                expert_down_entries: &moe.expert_down_entries,
                layer_idx: layer.layer as u16,
                norm_topk_prob: config.norm_topk_prob,
                row_gather: None,
                shared_down_out: None,
                tile_exchange: None,
            })
        }
        _ => {
            return Err(invalid(format!(
                "layer {} FFN weights and MoE runtime disagree",
                layer.layer
            )))
        }
    };
    Ok(Qwen4LayerDescription {
        attn_hyper: hyper_desc(weights, &layer.attn_hyper)?,
        mlp_hyper: hyper_desc(weights, &layer.mlp_hyper)?,
        attention,
        ffn,
    })
}

pub(crate) struct Qwen4ExpertView {
    gate_up: ProjectionView,
    down: ProjectionView,
}

struct ExpertViewSet<'a> {
    experts: &'a [Qwen4ExpertView],
    identity: u64,
    host_mapped: bool,
    tier_identity: Option<&'a hipfire_dispatch::families::moe::ExpertTierIdentity>,
}

impl RoutedExpertWeights for ExpertViewSet<'_> {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }

    fn immutable_identity(&self) -> Option<u64> {
        Some(self.identity)
    }

    fn host_mapped(&self) -> bool {
        self.host_mapped
    }

    fn tier_identity(&self) -> Option<&hipfire_dispatch::families::moe::ExpertTierIdentity> {
        self.tier_identity
    }
}

/// Source of [`Qwen4MoeLayerRuntime::identity`].
static NEXT_EXPERT_SET_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Architecture-owned sealed-MoE resources.  Pointer tables and expert views
/// are built once with the assembled resident weights and reused for every
/// token; no per-token resource or metadata allocation occurs.
pub(crate) struct Qwen4MoeLayerRuntime {
    table: ExpertTable,
    cache: ExpertBindingCache,
    router: ProjectionView,
    shared_scalar: ProjectionView,
    shared_gate: ProjectionView,
    shared_up: ProjectionView,
    shared_down: ProjectionView,
    /// `experts` is built once in `from_moe` and never mutated, so this
    /// construction-unique id is its [`RoutedExpertWeights::immutable_identity`].
    experts: Vec<Qwen4ExpertView>,
    identity: u64,
    experts_all_gate_up_mq4: bool,
    expert_gate_up_ptrs: GpuTensor,
    expert_down_ptrs: GpuTensor,
    /// Host entries the two tables above were uploaded from. Kept so dispatch can
    /// prove the table contents name the live expert tensors before a retained
    /// body records them.
    expert_gate_up_entries: Vec<usize>,
    expert_down_entries: Vec<usize>,
    /// Every routed QT44 gate/up and QT53 down header behind the two tables
    /// satisfies `zp == -8*sc`, checked on the device once after load
    /// ([`Self::verify_symmetric`]); only then may this layer declare
    /// [`MoeRouteCapability::Qt44Qt53GroupedSymmetric`]. False unless the
    /// opt-in route was requested.
    symmetric: bool,
    /// The layer's routed experts are the stock-FP8 recipe's qt=55
    /// `FP8E4M3B128` blocks (set by `build` from the immutable accessor
    /// tables). Such a layer always declares
    /// [`MoeRouteCapability::Fp8B128Grouped`], never runs the symmetric check
    /// and is sealed as grouped prefill at every row count, one row included.
    fp8: bool,
    /// Tier layers use `view.any_host()`. Legacy/EP layers report their
    /// local owning tensor's residency, never a dummy or peer view's bit.
    /// Residency changes weight-stream reuse, not each output's fold order.
    host_mapped: bool,
    /// Opt-in routed CPU splice, only for whole host-mapped trunk aggregates.
    pub(crate) cpu_deferred_experts: bool,
    cpu_expert_owners: Option<cpu_splice::CpuExpertOwners>,
    /// D2 split (the explicit `memory.offload_exec=cpu` over an expert hot
    /// set, or the per-load default over a hot-set tier or an un-tiered
    /// host-mapped aggregate): the COLD experts' retained CPU spans, one
    /// [`cpu_splice::HostSpanOwner`] per layer. Never set together with
    /// `cpu_deferred_experts` (distinct numerical mode).
    cpu_cold_owners: Option<cpu_splice::HostSpanOwner>,
    /// The layer's preallocated split executor (masked route + slot storage),
    /// released through `free_gpu` only.
    cpu_split: Option<hipfire_dispatch::cpu_split::CpuSplitExecutor>,
    /// Keeps the frozen payload owners and borrowed pointer tables alive.
    tier: Option<crate::expert_tier_owners::ExpertTierLease>,
    tier_identity: Option<hipfire_dispatch::families::moe::ExpertTierIdentity>,
    /// `Some` on a trunk layer of an expert-slab artifact: the layer's experts
    /// are views into its one slab arena with a codec each.
    slab: Option<Qwen4SlabLayer>,
    /// The layer's label (`layer.layer`): what its sealed calls carry as
    /// `layer_idx`, and so what its retained-route identities name.
    label: usize,
    /// The layer's retained-route `(key, identity)` pairs, derived once after
    /// the live binding (never per token).
    retained_identities: Vec<(u64, String)>,
    expert_keep_mask: Option<GpuTensor>,
}

/// Per-layer state of a sealed expert-slab layer ([`crate::expert_arena`]).
pub(crate) struct Qwen4SlabLayer {
    /// The layer's cold codec id on this device (W3 = 1, E8P = 3; `0` = MQ4,
    /// no cold bucket, for an MQ4-only layer): decided once at load from the
    /// validated span table ([`crate::expert_arena::ExpertArenaLayer::cold_codec`]),
    /// what every K1 / tile-mask launch of the layer binds to.
    cold_codec: u8,
    /// Real codec byte of every expert (`0` = MQ4, `1` = W3, `3` = E8P), from
    /// the validated span table; never inferred from any single expert. The
    /// device planner table is NOT this: it is membership only
    /// ([`slab_codec_tag_bytes`]).
    codecs: Vec<u8>,
    /// Per-expert gate/up and down dtypes (parallel to `codecs`).
    gate_up_dtypes: Vec<DType>,
    down_dtypes: Vec<DType>,
    /// Mixed layers only: the layer's dtype-tag tensor ([`slab_codec_tag_bytes`]):
    /// `[512]` u8 planner membership (cold = 1, MQ4 = 0) followed by `[512]`
    /// i32 real codecs (`hipfire_dispatch::pipeline::slab_route`).
    codec_tags: Option<GpuTensor>,
    /// Base device address of the layer's arena and its byte length: part of the
    /// layer's arena generation (replay keys).
    arena_base: usize,
    arena_bytes: usize,
}

impl Qwen4SlabLayer {
    pub(crate) fn mixed(&self) -> bool {
        self.codec_tags.is_some()
    }
}

impl Qwen4MoeLayerRuntime {
    /// The layer's routed experts live in pinned host memory (every expert of
    /// the layer: S1 places whole layers).
    pub(crate) fn s1_host_mapped(&self) -> bool {
        self.host_mapped
    }

    /// Whether this layer is the qualified S1 shape: uniform MQ4 experts with
    /// no tier split, no stock FP8, no expert-keep
    /// pruning mask and no non-MQ4 slab codec. `Err` names the first reason;
    /// the S1 selector turns it into a pre-mutation incumbent verdict.
    /// S1 never runs host arithmetic: D1 is switched off per S1 tile binding
    /// (`s1_tile_cpu_deferred`) and D2 is bound splitless
    /// (`routed_view(false, ..)`), so `offload_exec` (D1 splice, explicit D2)
    /// does not affect admission. Fp8, tier layers, non-uniform MQ4, the
    /// keep mask and a mixed slab stay refused.
    pub(crate) fn s1_layer_check(&self) -> Result<(), &'static str> {
        if self.fp8 {
            return Err("stock FP8 experts");
        }
        if self.tier.is_some() {
            return Err("expert tiers");
        }
        if !self.experts_all_gate_up_mq4 {
            return Err("experts are not uniform MQ4");
        }
        if self.expert_keep_mask.is_some() {
            return Err("expert-keep pruning mask");
        }
        if let Some(slab) = self.slab.as_ref() {
            if slab
                .codecs
                .iter()
                .any(|codec| *codec != crate::expert_arena::CODEC_MQ4)
            {
                return Err("mixed-codec slab layer");
            }
        }
        if self.expert_gate_up_entries.len() != self.experts.len()
            || self.expert_down_entries.len() != self.experts.len()
        {
            return Err("expert table entries differ from the expert views");
        }
        Ok(())
    }

    /// The layer's device pointer tables (global expert id indexed u64, gate/up
    /// then down): the binding of an S1 resident layer.
    pub(crate) fn s1_tables(&self) -> (&GpuTensor, &GpuTensor) {
        (&self.expert_gate_up_ptrs, &self.expert_down_ptrs)
    }

    /// Experts one table holds (the model's expert count).
    pub(crate) fn s1_num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Per-expert payload bytes `(gate/up, down)` of expert `id`, `None` for an
    /// expert the layer does not hold.
    pub(crate) fn s1_expert_bytes(&self, id: usize) -> Option<(usize, usize)> {
        let expert = self.experts.get(id)?;
        if expert.gate_up.tensor.buf.size() == 0 || expert.down.tensor.buf.size() == 0 {
            return None;
        }
        Some((
            crate::projection::checked_bytes(expert.gate_up.m, expert.gate_up.row_stride)?,
            crate::projection::checked_bytes(expert.down.m, expert.down.row_stride)?,
        ))
    }

    /// `(gate/up, down)` dtypes of expert `id`, `None` for an expert the layer
    /// does not hold: what the SP admission reads to prove the CPU engine can
    /// execute the host layers' experts.
    pub(crate) fn s1_expert_dtypes(&self, id: usize) -> Option<(DType, DType)> {
        let expert = self.experts.get(id)?;
        if expert.gate_up.tensor.buf.size() == 0 || expert.down.tensor.buf.size() == 0 {
            return None;
        }
        Some((expert.gate_up.dtype, expert.down.dtype))
    }

    /// The transport sources of this host layer's experts: pinned CPU
    /// addresses of the immutable weight payload, translated from the device
    /// alias addresses of the layer's pointer entries. A slab arena layer has
    /// one host range; the aggregate packed layer has one per projection.
    pub(crate) fn s1_host_sources(
        &self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        layer: usize,
    ) -> Result<panel_stage::LayerExpertSources, Qwen4GpuForwardError> {
        use panel_stage::{AliasExpert, LayerExpertSources, PanelCodec};
        self.s1_layer_check().map_err(|why| {
            invalid(format!(
                "slow-link layer {layer} is not S1 qualified: {why}"
            ))
        })?;
        if !self.host_mapped {
            return Err(invalid(format!(
                "slow-link layer {layer} has no host-mapped experts to stream"
            )));
        }
        let mut aliases: Vec<Option<AliasExpert>> = Vec::with_capacity(self.experts.len());
        for id in 0..self.experts.len() {
            aliases.push(
                self.s1_expert_bytes(id)
                    .map(|(gate_up_len, down_len)| AliasExpert {
                        codec: PanelCodec(0),
                        gate_up_addr: self.expert_gate_up_entries[id],
                        gate_up_len,
                        down_addr: self.expert_down_entries[id],
                        down_len,
                    }),
            );
        }
        let weights = bundle
            .weights
            .layer(layer)
            .ok_or_else(|| invalid(format!("slow-link layer {layer} is not owned")))?
            .ffn
            .moe()
            .ok_or_else(|| invalid(format!("slow-link layer {layer} has no MoE weights")))?;
        // Each pinned owner's range is paired with a PHYSICAL keepalive: a lease
        // that holds the `hipHostMalloc` allocation until the last copy source
        // drops, whatever happens to the weights, the registry or the `Gpu`.
        fn leased_range(
            gpu: &mut Gpu,
            tensor: &GpuTensor,
            layer: usize,
        ) -> Result<((usize, usize, usize), rdna_compute::HostAllocLease), Qwen4GpuForwardError>
        {
            let lease = gpu.lease_host_owner(tensor)?;
            let host = gpu.host_bytes(tensor).ok_or_else(|| {
                invalid(format!(
                    "slow-link layer {layer}: a host expert owner has no pinned host bytes"
                ))
            })?;
            if host.as_ptr() as usize != lease.host_ptr() as usize
                || lease.device_alias() != tensor.buf.as_ptr() as usize
            {
                return Err(invalid(format!(
                    "slow-link layer {layer}: the host lease names another allocation than its owner tensor"
                )));
            }
            Ok((
                (host.as_ptr() as usize, lease.device_alias(), host.len()),
                lease,
            ))
        }
        // SAFETY: each range is the pinned host allocation of the layer's own
        // resident weight tensor, immutable for as long as the lease (the
        // sources' `owner`) lives; every extent is bounds checked against its
        // range by the constructor.
        match (self.slab.as_ref(), weights.expert_arena.as_ref()) {
            (Some(slab), Some(arena_ref)) => {
                let tensor = bundle.weights.resident(arena_ref)?;
                let ((cpu_base, alias_base, host_len), lease) = leased_range(gpu, tensor, layer)?;
                if host_len < slab.arena_bytes || alias_base != slab.arena_base {
                    return Err(invalid(format!(
                        "slow-link layer {layer}: host slab arena does not match its plan"
                    )));
                }
                let owner = lease.keepalive();
                unsafe {
                    LayerExpertSources::from_alias(
                        layer,
                        cpu_base,
                        slab.arena_base,
                        slab.arena_bytes,
                        &aliases,
                        owner,
                    )
                }
            }
            _ => {
                let (gate_up, gate_up_lease) = leased_range(
                    gpu,
                    bundle.weights.resident(&weights.experts_gate_up)?,
                    layer,
                )?;
                let (down, down_lease) =
                    leased_range(gpu, bundle.weights.resident(&weights.experts_down)?, layer)?;
                let owner: Arc<dyn std::any::Any + Send + Sync> =
                    Arc::new((gate_up_lease, down_lease));
                unsafe {
                    LayerExpertSources::from_alias_split(layer, gate_up, down, &aliases, owner)
                }
            }
        }
    }
}

/// Host bytes of a mixed slab layer's tag tensor from its real per-expert
/// codecs. The K1 union planner reads the leading `[512]` u8 table as
/// bucket membership only (bucket 0 = cold, 1 = MQ4: byte `1` selects the
/// cold bucket, `0` MQ4), so the layer's cold codec (W3 or E8P) is written as
/// `1` and MQ4 as `0`; the table-driven tile mask reads the trailing `[512]`
/// i32 table, which keeps the real codec of each expert (0 / 1 / 3).
/// A layer holds MQ4 plus one cold codec ([`rdna_compute::qwen4_slab_layer_cold_codec`]),
/// so membership is unambiguous.
fn slab_codec_tag_bytes(codecs: &[u8]) -> Vec<u8> {
    use crate::expert_arena::CODEC_MQ4;
    let mut bytes =
        Vec::with_capacity(hipfire_dispatch::pipeline::slab_route::CODEC_TAG_ALLOC_BYTES);
    bytes.extend(codecs.iter().map(|codec| u8::from(*codec != CODEC_MQ4)));
    for codec in codecs {
        bytes.extend_from_slice(&i32::from(*codec).to_le_bytes());
    }
    bytes
}

pub(crate) use cpu_splice::{ColdLayerRefs, ColdLayerSource};
mod cpu_slab_owner;
mod cpu_splice;
pub(crate) mod ep_reserve_forward;
pub use ep_reserve_forward::ep_reserve_receipt_scope;
pub(crate) mod ep_reserve_loan;
pub(crate) mod ep_reserve_scratch;

impl RoutedExpertWeights for Qwen4MoeLayerRuntime {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }

    fn immutable_identity(&self) -> Option<u64> {
        Some(self.identity)
    }

    fn host_mapped(&self) -> bool {
        self.host_mapped
    }

    fn cpu_splice_expert(
        &self,
        _gpu: &Gpu,
        expert_idx: usize,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        let expert = self.experts.get(expert_idx)?;
        // Tier split: only cold (host-placed) experts resolve, from the
        // retained host owner's CPU address, never the device alias.
        if let Some(owners) = self.cpu_cold_owners.as_ref() {
            return owners.payload(expert_idx, expert.gate_up.dtype, expert.down.dtype);
        }
        self.cpu_expert_owners.as_ref()?.payload(
            expert_idx,
            expert.gate_up.dtype,
            expert.down.dtype,
        )
    }

    fn cpu_split(&self) -> Option<&hipfire_dispatch::cpu_split::CpuSplitExecutor> {
        self.cpu_split.as_ref()
    }

    fn tier_identity(&self) -> Option<&hipfire_dispatch::families::moe::ExpertTierIdentity> {
        self.tier_identity.as_ref()
    }
}

/// A layer's runtime seen through a window the D2 split must not serve: the
/// same experts, tables and identity, but no split executor. The D2 executor
/// gate keys on the row count alone, so the prefill-side caller binds this view
/// instead of the runtime for any call that
/// [`hipfire_dispatch::cpu_split::cpu_splice_window`] rejects.
#[repr(transparent)]
struct SplitlessExperts(Qwen4MoeLayerRuntime);

impl RoutedExpertWeights for SplitlessExperts {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.0.get(expert_idx)
    }

    fn immutable_identity(&self) -> Option<u64> {
        self.0.immutable_identity()
    }

    fn host_mapped(&self) -> bool {
        self.0.host_mapped()
    }

    fn cpu_splice_expert(
        &self,
        gpu: &Gpu,
        expert_idx: usize,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        self.0.cpu_splice_expert(gpu, expert_idx)
    }

    fn tier_identity(&self) -> Option<&hipfire_dispatch::families::moe::ExpertTierIdentity> {
        self.0.tier_identity()
    }

    // Every other trait method is forwarded so the prefill binding is identical
    // to the runtime's in every capability except the split executor.
    fn host_mapped_projections(&self) -> (bool, bool) {
        self.0.host_mapped_projections()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Qwen4MoeLayerRuntime {
    /// The weights binding for a call: this runtime when the D2 split may serve
    /// it (or none is attached), otherwise the executor-less view. `d2_phase`
    /// is false inside a prefill tile (see [`d2_executor_admitted`]).
    /// `lab_armed` is the existing-oracle-override presence for this call
    /// (env or runtime override on an exact oracle arch with an eager verify
    /// scope): only presence with allocated executor capacity admits a
    /// 5..64-row window — never the `few_row_limit` value as a capacity (that
    /// is a GPU projection-route limit, not CPU storage). `verify_rows` is the
    /// frozen row-exact width, an upper clamp only. Ordinary prefill and lane
    /// callers pass `false` and keep the default-4 gate.
    pub(crate) fn routed_view(
        &self,
        d2_phase: bool,
        rows: usize,
        route_rows: usize,
        lab_armed: bool,
        verify_rows: usize,
    ) -> &dyn RoutedExpertWeights {
        let capacity = self
            .cpu_split
            .as_ref()
            .map(|executor| executor.capacity_rows())
            .unwrap_or(0);
        if d2_executor_admitted(
            self.cpu_split.is_some(),
            d2_phase,
            rows,
            route_rows,
            lab_armed,
            verify_rows,
            capacity,
        ) {
            return self;
        }
        // The view must outlive the sealed step list that borrows it, so it
        // cannot be a local wrapper (a per-forward arena of wrappers would be
        // an avoidable allocation on the prefill path); a `#[repr(transparent)]`
        // newtype makes the cast a no-op.
        // SAFETY: `SplitlessExperts` is `#[repr(transparent)]` over
        // `Qwen4MoeLayerRuntime`, so the two references have identical layout,
        // alignment and validity, and the view borrows `self` for its lifetime.
        unsafe { &*(self as *const Self as *const SplitlessExperts) }
    }
}

/// The runtime one layer's call binds: a balanced live runtime (gathered EP
/// few-row forward), else for an EP forward's full chunk (`full_chunk`) the
/// static runtime under a tile route or the EP full table (`ep_full`: `None`
/// outside EP, `Some(layer's full runtime)` under EP), else the static runtime.
pub(crate) fn select_layer_moe<'a, R>(
    balanced_live: Option<&'a R>,
    ep_full: Option<Option<&'a R>>,
    full_chunk: bool,
    tile_route: bool,
    static_moe: Option<&'a R>,
) -> Option<&'a R> {
    match (balanced_live, ep_full) {
        (Some(live), _) => Some(live),
        (None, Some(full)) if full_chunk => {
            if tile_route {
                static_moe
            } else {
                full
            }
        }
        _ => static_moe,
    }
}

/// The routed-expert binding a MoE seal receives: `view` of the runtime the
/// layer selected, never of another runtime (the layer's table and cache are
/// bound to the selected one). Without a selected runtime the description's
/// own binding stands.
pub(crate) fn bind_routed_view<'a, R>(
    selected: Option<&'a R>,
    bound: &'a dyn RoutedExpertWeights,
    view: impl FnOnce(&'a R) -> &'a dyn RoutedExpertWeights,
) -> &'a dyn RoutedExpertWeights {
    selected.map_or(bound, view)
}

/// The runtime one gathered EP lane MoE invocation binds: exactly the
/// singleton's selection for a 1..=[`QWEN4_EP_GATHER_MAX_ROWS`]-row gathered
/// forward (`full_chunk = false`, no tile route): the layer's balanced live
/// runtime, else the static runtime; never the EP full table. Outside EP
/// (`balanced_live` and `ep_full` absent) it is the static runtime.
pub(crate) fn select_ep_lane_moe<'a, R>(
    balanced_live: Option<&'a R>,
    ep_full: Option<Option<&'a R>>,
    static_moe: Option<&'a R>,
) -> Option<&'a R> {
    select_layer_moe(balanced_live, ep_full, false, false, static_moe)
}

/// Whether a gathered EP MoE call of `rows` rows binds a separate
/// shared-down output: the singleton's few-row forward (`rows >= 2`) does, its
/// one-row decode does not.
pub(crate) const fn ep_lane_shared_down(rows: usize) -> bool {
    rows >= 2
}

/// A lane MoE invocation under EP runs the singleton's gathered forward: its
/// real rows are its routing rows, at most [`QWEN4_EP_GATHER_MAX_ROWS`]. Wider
/// or re-routed calls would be the full-table chunk route, which a lane step
/// does not run.
pub(crate) fn ep_lane_moe_rows(rows: usize, route_rows: usize) -> Result<(), Qwen4GpuForwardError> {
    if rows == 0 {
        return Err(invalid(
            "an EP lane MoE invocation carries at least one row",
        ));
    }
    if rows > QWEN4_EP_GATHER_MAX_ROWS {
        return Err(invalid(format!(
            "an EP lane MoE invocation of {rows} rows exceeds the {QWEN4_EP_GATHER_MAX_ROWS}-row gather cap"
        )));
    }
    if route_rows != rows {
        return Err(invalid(format!(
            "an EP lane MoE invocation of {rows} rows routes {route_rows} rows; the gathered forward needs them equal"
        )));
    }
    Ok(())
}

/// The gather buffer (`0` or `1`, the trunk buffers) of the `ordinal`-th MoE
/// invocation of a lane step, counted over the whole step in emission order.
/// Consecutive invocations never share a buffer (`ep_row_gather` hazard rule),
/// whatever the lane/layer interleaving.
pub(crate) const fn ep_lane_buffer(ordinal: usize) -> usize {
    ordinal % 2
}

/// A step's MoE invocation count must be even so the next step's first
/// invocation (ordinal 0, buffer 0) never repeats this step's last buffer.
pub(crate) fn ep_lane_check_invocations(invocations: usize) -> Result<(), Qwen4GpuForwardError> {
    if invocations % 2 != 0 {
        return Err(invalid(format!(
            "an EP lane step emitted {invocations} MoE invocations; an odd count would reuse a gather buffer across consecutive steps"
        )));
    }
    Ok(())
}

/// Install the singleton's gathered-forward EP binding on one lane MoE
/// binding: the row gather, and for `rows >= 2` the singleton scratch's
/// `moe_down_expanded` as the unfused shared-down output.
pub(crate) fn bind_ep_lane_gather<'a>(
    binding: &mut Qwen4MoeBinding<'a>,
    rows: usize,
    gather: &'a dyn MoeRowGather,
    shared_down: &'a GpuTensor,
) {
    binding.row_gather = Some(gather);
    if ep_lane_shared_down(rows) {
        binding.shared_down_out = Some(shared_down);
    }
}

/// What a lane step borrows from an EP forward: the installed row gather, the
/// two trunk gather buffers the step alternates over, and the singleton
/// scratch's `moe_down_expanded` (the shared-down output of few-row calls).
pub(crate) struct Qwen4EpLane<'a> {
    pub(crate) gather: &'a dyn MoeRowGather,
    pub(crate) buffers: [&'a GpuTensor; 2],
    pub(crate) shared_down: &'a GpuTensor,
}

/// The runtime a lane MoE invocation of `layer` binds (see
/// [`select_ep_lane_moe`]); the static runtime outside EP. Takes the forward's
/// `ep` and `moe` fields (`fwd.ep.as_ref()`, `&fwd.moe`) rather than the whole
/// forward, so the lane step can keep the result alive while it mutably uses
/// other fields.
pub(crate) fn lane_layer_moe<'a>(
    ep: Option<&'a Qwen4ForwardEp>,
    moe: &'a [Option<Qwen4MoeLayerRuntime>],
    layer: usize,
) -> Option<&'a Qwen4MoeLayerRuntime> {
    let balanced_live = ep
        .and_then(|ep| ep.balanced.get(layer))
        .and_then(Option::as_ref)
        .map(|balanced| &balanced.live);
    select_ep_lane_moe(
        balanced_live,
        ep.map(|ep| ep.moe_full.get(layer)),
        moe.get(layer).and_then(Option::as_ref),
    )
}

/// How an EP lane step binds one layer's routed experts (see
/// [`Qwen4GpuForward::lane_ep_expert_kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpLaneExpertKind {
    /// The static EP decode runtime over a fully resident packed aggregate.
    StaticResidentPacked,
    /// The live balanced-ownership runtime over a host-mapped packed aggregate.
    BalancedHostMappedPacked,
}

/// What [`classify_ep_lane_expert`] needs of one layer's [`Qwen4MoeLayerRuntime`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EpLaneRuntimeFacts {
    /// The runtime's layer label.
    label: usize,
    host_mapped: bool,
    /// Non-empty expert views, every one QT44 gate/up + QT53 down.
    packed_mq4: bool,
    /// No CPU-deferred/split/cold owner, tier, slab, stock FP8 or keep mask.
    plain: bool,
}

/// Classify one EP layer's lane expert binding from plain facts: the row gather
/// is installed, `selected` (what [`lane_layer_moe`] picked; `balanced_selected`
/// when that is the layer's live balanced runtime) and the layer's `static_moe`
/// runtime are both labelled for `layer` and plain packed QT44/QT53, and the
/// residency agrees with the selection: a live balanced runtime over a
/// host-mapped layer, else the static runtime over a resident layer. Anything
/// else (a host-mapped layer without a live balanced runtime, a balanced
/// selection over a resident layer, a missing runtime) is `None`.
fn classify_ep_lane_expert(
    layer: usize,
    gather_installed: bool,
    balanced_selected: bool,
    selected: Option<EpLaneRuntimeFacts>,
    static_moe: Option<EpLaneRuntimeFacts>,
) -> Option<EpLaneExpertKind> {
    if !gather_installed {
        return None;
    }
    let (selected, static_moe) = (selected?, static_moe?);
    for runtime in [selected, static_moe] {
        if runtime.label != layer || !runtime.packed_mq4 || !runtime.plain {
            return None;
        }
    }
    match (
        balanced_selected,
        selected.host_mapped,
        static_moe.host_mapped,
    ) {
        (true, true, true) => Some(EpLaneExpertKind::BalancedHostMappedPacked),
        (false, false, false) => Some(EpLaneExpertKind::StaticResidentPacked),
        _ => None,
    }
}

/// The EP lane resources from the forward's `ep` and `scratch` fields
/// (`fwd.ep.as_ref()`, `&fwd.scratch`): `None` outside EP, an error under EP
/// before `attach_ep` installed the row gather.
pub(crate) fn ep_lane<'a>(
    ep: Option<&'a Qwen4ForwardEp>,
    scratch: &'a Qwen4GpuForwardScratch,
) -> Result<Option<Qwen4EpLane<'a>>, Qwen4GpuForwardError> {
    let Some(ep) = ep else {
        return Ok(None);
    };
    let gather = ep
        .gather
        .as_deref()
        .ok_or_else(|| invalid("EP lane step before attach_ep: the row gather is not installed"))?;
    Ok(Some(Qwen4EpLane {
        gather: gather as &dyn MoeRowGather,
        buffers: [&ep.buffers[0], &ep.buffers[1]],
        shared_down: &scratch.moe_down_expanded,
    }))
}
/// Whether a layer's D2 executor is bound for one MoE call: always when no
/// executor is attached (nothing to withhold), otherwise only in a non-prefill
/// phase (decode, verify; `d2_phase`) **and** for a whole-routing-window call
/// inside the qualified width: at most 4 rows on the default route, or a whole
/// logical 1..64-row window inside an explicit lab-wide scope (existing oracle
/// override present + eager verify) when the executor's allocated
/// `capacity_rows` covers it. `lab_armed` is that presence signal — never the
/// `few_row_limit` value (a GPU route limit, not CPU storage). A prompt of
/// 1..=4 rows inside a prefill tile, a 1..=4-row tail of a long prefill and
/// every unqualified wider window take the GPU route. No chunking: an admitted
/// wide window runs one forward with full-window state/capture.
fn d2_executor_admitted(
    has_split: bool,
    d2_phase: bool,
    rows: usize,
    route_rows: usize,
    lab_armed: bool,
    verify_rows: usize,
    capacity_rows: usize,
) -> bool {
    if !has_split {
        return true;
    }
    if !d2_phase {
        return false;
    }
    let scope = hipfire_dispatch::families::moe::CpuWideVerifyScope::from_lab_presence(
        lab_armed,
        verify_rows,
    );
    hipfire_dispatch::families::moe::cpu_wide_window_qualified(
        true,
        rows,
        route_rows,
        scope,
        capacity_rows,
    )
}

/// Whether a load runs host arithmetic on **every** call: a D1 whole-layer
/// splice layer, or a D2 split requested by the explicit
/// `memory.offload_exec=cpu`. Only such a load caps speculative width (verify,
/// replay, MTP `k`, n-gram take, lane verify) and needs eager HIP for the whole
/// forward. The per-load measured D2 default is not load-wide: it executes per
/// call ([`host_arith_call`]), so an automatic-only load keeps every width and
/// eager rule of an unarmed load.
const fn host_arith_every_call(d1: bool, has_split: bool, split_explicit: bool) -> bool {
    d1 || (has_split && split_explicit)
}

/// Whether one call runs host arithmetic: `every_call` (a load
/// [`host_arith_every_call`] holds for), or an armed D2 split (`has_split`) on
/// a whole qualified non-prefill window ([`d2_executor_admitted`]): at most 4
/// rows by default, or a whole 1..64-row window inside an explicit lab-wide
/// scope (override presence + eager verify) with allocated capacity.
fn host_arith_call(
    every_call: bool,
    has_split: bool,
    prefill: bool,
    rows: usize,
    route_rows: usize,
    lab_armed: bool,
    verify_rows: usize,
    capacity_rows: usize,
) -> bool {
    every_call
        || (has_split
            && d2_executor_admitted(
                true,
                !prefill,
                rows,
                route_rows,
                lab_armed,
                verify_rows,
                capacity_rows,
            ))
}

/// Gather buffers one EP rank owns: parity `L % 2` over the trunk layers and
/// parity `2 + j % 2` for the MTP head, each `QWEN4_EP_GATHER_MAX_ROWS *
/// num_experts_per_tok * hidden` F32.
pub const QWEN4_EP_GATHER_BUFFERS: usize = 4;

/// Most rows a few-row forward (verify / replay / takeover) may carry through
/// the EP table + row gather; larger chunks run the full (P2P) table.
pub const QWEN4_EP_GATHER_MAX_ROWS: usize = 8;

/// Which EP gather buffer a MoE invocation uses: trunk layer `L` uses
/// `L % 2`; the MTP head (a layer index at or past the trunk layer count)
/// uses `2 + (layer_idx - trunk_layers) % 2`.
pub(crate) fn qwen4_ep_buffer_index(layer_idx: usize, trunk_layers: usize) -> usize {
    if layer_idx >= trunk_layers {
        2 + (layer_idx - trunk_layers) % 2
    } else {
        layer_idx % 2
    }
}

/// Device pointers and byte lengths of one rank's compact routed-expert
/// tensors of one MoE layer (`local_count(512)` experts of the owning rank, local
/// index `e / ranks`).
/// The same shape describes the per-rank zero dummy expert (one expert).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4PeerExpertSource {
    pub gate_up_base: usize,
    pub gate_up_bytes: usize,
    pub down_base: usize,
    pub down_bytes: usize,
}

/// How one rank's per-layer expert pointer table is populated.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Qwen4ExpertTableLayout<'a> {
    /// Every expert from the layer's resident tensors (one device).
    Single,
    /// Frozen absolute pointers into the two model-level tier owners.
    Tier {
        gate_up: &'a [usize],
        down: &'a [usize],
    },
    /// EP decode table: owned experts name the compact local tensors,
    /// every other expert names the rank's zero dummy expert.
    Ep {
        placement: Qwen4EpPlacement,
        dummy: Qwen4PeerExpertSource,
    },
    /// EP chunk table: owned experts name the local compact tensors, every
    /// other expert names the owning peer's compact tensors (P2P), `peers`
    /// indexed by rank.
    Full {
        placement: Qwen4EpPlacement,
        peers: &'a [Qwen4PeerExpertSource],
    },
}

/// The allocation one expert-table entry names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4ExpertOrigin {
    /// This rank's resident (compact) expert tensors.
    Local,
    /// A checked projection in one of the immutable model-level owners.
    Tier,
    /// This rank's zero dummy expert.
    Dummy,
    /// The compact expert tensors of the named peer rank.
    Peer(usize),
}

/// Host pointer entries of one expert in the gate/up and down tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Qwen4ExpertEntry {
    pub(crate) origin: Qwen4ExpertOrigin,
    pub(crate) gate_up_ptr: usize,
    pub(crate) down_ptr: usize,
}

fn validate_ep_placement(
    placement: Qwen4EpPlacement,
    num_experts: usize,
) -> Result<(), Qwen4GpuForwardError> {
    if placement.ranks == 0 || placement.rank >= placement.ranks {
        return Err(invalid(format!(
            "EP placement rank {} is outside {} ranks",
            placement.rank, placement.ranks
        )));
    }
    if num_experts < placement.ranks {
        return Err(invalid(format!(
            "{num_experts} experts cannot cover {} EP ranks",
            placement.ranks
        )));
    }
    Ok(())
}

impl Qwen4ExpertTableLayout<'_> {
    /// Experts the layer's own resident tensors hold under this layout.
    fn local_experts(&self, num_experts: usize) -> Result<usize, Qwen4GpuForwardError> {
        match self {
            Self::Single | Self::Tier { .. } => Ok(num_experts),
            Self::Ep { placement, .. } | Self::Full { placement, .. } => {
                validate_ep_placement(*placement, num_experts)?;
                Ok(placement.local_count(num_experts))
            }
        }
    }
}

/// Device address of expert slot `index` of a tensor `(base, total)` bytes
/// long holding `expert_bytes` per slot; the whole slot must lie inside the
/// tensor.
fn expert_slot_ptr(
    what: &str,
    base: usize,
    total: usize,
    index: usize,
    expert_bytes: usize,
) -> Result<usize, Qwen4GpuForwardError> {
    let offset = index
        .checked_mul(expert_bytes)
        .ok_or_else(|| invalid(format!("{what} expert {index} offset overflows")))?;
    let end = offset
        .checked_add(expert_bytes)
        .ok_or_else(|| invalid(format!("{what} expert {index} end overflows")))?;
    if base == 0 {
        return Err(invalid(format!("{what} base pointer is null")));
    }
    if end > total {
        return Err(invalid(format!(
            "{what} expert slot {index} [{offset}, {end}) exceeds the {total}-byte tensor"
        )));
    }
    base.checked_add(offset)
        .ok_or_else(|| invalid(format!("{what} expert {index} address overflows")))
}

/// Prove both the logical owner-relative span and its absolute table address.
/// This never derives an owner base from an expert pointer.
fn tier_expert_extent(
    what: &str,
    owner: Option<(usize, usize)>,
    offset: u64,
    len: u64,
    expected_len: usize,
    pointer: usize,
) -> Result<(), Qwen4GpuForwardError> {
    let (base, bytes) = owner.ok_or_else(|| invalid(format!("{what}: missing tier owner")))?;
    let offset =
        usize::try_from(offset).map_err(|_| invalid(format!("{what}: offset overflow")))?;
    let len = usize::try_from(len).map_err(|_| invalid(format!("{what}: length overflow")))?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid(format!("{what}: extent overflow")))?;
    let address = base
        .checked_add(offset)
        .ok_or_else(|| invalid(format!("{what}: address overflow")))?;
    if base == 0
        || len != expected_len
        || end > bytes
        || base.checked_add(bytes).is_none()
        || pointer != address
        || pointer % 256 != 0
    {
        return Err(invalid(format!(
            "{what}: tier span {offset}+{len} / {bytes} or pointer {pointer:#x} is invalid"
        )));
    }
    Ok(())
}

/// Pure arithmetic behind every expert pointer table: the host entry of each
/// global expert under `layout`. `local` names this rank's own resident
/// tensors (all experts under `Single`, the compact
/// `placement.local_count(num_experts)` under EP). Nothing here touches a device.
pub(crate) fn qwen4_expert_table_entries(
    layout: &Qwen4ExpertTableLayout<'_>,
    local: Qwen4PeerExpertSource,
    num_experts: usize,
    gate_up_expert_bytes: usize,
    down_expert_bytes: usize,
) -> Result<Vec<Qwen4ExpertEntry>, Qwen4GpuForwardError> {
    let mut entries = Vec::with_capacity(num_experts);
    match layout {
        Qwen4ExpertTableLayout::Single => {
            for expert in 0..num_experts {
                entries.push(Qwen4ExpertEntry {
                    origin: Qwen4ExpertOrigin::Local,
                    gate_up_ptr: expert_slot_ptr(
                        "gate/up",
                        local.gate_up_base,
                        local.gate_up_bytes,
                        expert,
                        gate_up_expert_bytes,
                    )?,
                    down_ptr: expert_slot_ptr(
                        "down",
                        local.down_base,
                        local.down_bytes,
                        expert,
                        down_expert_bytes,
                    )?,
                });
            }
        }
        Qwen4ExpertTableLayout::Tier { gate_up, down } => {
            if gate_up.len() != num_experts || down.len() != num_experts {
                return Err(invalid("tier pointer table has the wrong expert count"));
            }
            for (&gate_up_ptr, &down_ptr) in gate_up.iter().zip(*down) {
                if gate_up_ptr == 0 || down_ptr == 0 {
                    return Err(invalid("tier pointer table contains a null expert"));
                }
                entries.push(Qwen4ExpertEntry {
                    origin: Qwen4ExpertOrigin::Tier,
                    gate_up_ptr,
                    down_ptr,
                });
            }
        }
        Qwen4ExpertTableLayout::Ep { placement, dummy } => {
            let per_rank = layout.local_experts(num_experts)?;
            check_exact_compact(local, per_rank, gate_up_expert_bytes, down_expert_bytes)?;
            if dummy.gate_up_bytes != gate_up_expert_bytes || dummy.down_bytes != down_expert_bytes
            {
                return Err(invalid(format!(
                    "EP dummy expert holds {}+{} bytes, one expert is {gate_up_expert_bytes}+{down_expert_bytes}",
                    dummy.gate_up_bytes, dummy.down_bytes
                )));
            }
            for expert in 0..num_experts {
                entries.push(if placement.owns(expert) {
                    let slot = placement.local_index(expert);
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Local,
                        gate_up_ptr: expert_slot_ptr(
                            "gate/up",
                            local.gate_up_base,
                            local.gate_up_bytes,
                            slot,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "down",
                            local.down_base,
                            local.down_bytes,
                            slot,
                            down_expert_bytes,
                        )?,
                    }
                } else {
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Dummy,
                        gate_up_ptr: expert_slot_ptr(
                            "dummy gate/up",
                            dummy.gate_up_base,
                            dummy.gate_up_bytes,
                            0,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "dummy down",
                            dummy.down_base,
                            dummy.down_bytes,
                            0,
                            down_expert_bytes,
                        )?,
                    }
                });
            }
        }
        Qwen4ExpertTableLayout::Full { placement, peers } => {
            let per_rank = layout.local_experts(num_experts)?;
            check_exact_compact(local, per_rank, gate_up_expert_bytes, down_expert_bytes)?;
            if peers.len() != placement.ranks {
                return Err(invalid(format!(
                    "EP full table needs {} peer sources, got {}",
                    placement.ranks,
                    peers.len()
                )));
            }
            if peers[placement.rank] != local {
                return Err(invalid(format!(
                    "EP peer source of rank {} does not name this rank's own expert tensors",
                    placement.rank
                )));
            }
            // Each peer's compact tensors hold exactly that rank's own count
            // (uneven splits: the low ranks hold one more).
            for (peer_rank, peer) in peers.iter().enumerate() {
                let peer_count = Qwen4EpPlacement {
                    rank: peer_rank,
                    ranks: placement.ranks,
                }
                .local_count(num_experts);
                check_exact_compact(*peer, peer_count, gate_up_expert_bytes, down_expert_bytes)?;
            }
            for expert in 0..num_experts {
                let owner = expert % placement.ranks;
                let slot = expert / placement.ranks;
                entries.push(if owner == placement.rank {
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Local,
                        gate_up_ptr: expert_slot_ptr(
                            "gate/up",
                            local.gate_up_base,
                            local.gate_up_bytes,
                            slot,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "down",
                            local.down_base,
                            local.down_bytes,
                            slot,
                            down_expert_bytes,
                        )?,
                    }
                } else {
                    let peer = peers[owner];
                    Qwen4ExpertEntry {
                        origin: Qwen4ExpertOrigin::Peer(owner),
                        gate_up_ptr: expert_slot_ptr(
                            "peer gate/up",
                            peer.gate_up_base,
                            peer.gate_up_bytes,
                            slot,
                            gate_up_expert_bytes,
                        )?,
                        down_ptr: expert_slot_ptr(
                            "peer down",
                            peer.down_base,
                            peer.down_bytes,
                            slot,
                            down_expert_bytes,
                        )?,
                    }
                });
            }
        }
    }
    Ok(entries)
}

/// `expert` is outside the layer's `qwen4.expert_keep` row; dense layers
/// (`ids == None`) have no absent experts.
fn expert_pruned(ids: Option<&[u16]>, expert: usize) -> bool {
    ids.is_some_and(|ids| u16::try_from(expert).is_ok_and(|e| ids.binary_search(&e).is_err()))
}

fn pruned_expert_entries(
    layout: &Qwen4ExpertTableLayout<'_>,
    local: Qwen4PeerExpertSource,
    experts: usize,
    ids: &[u16],
    gu_bytes: usize,
    dn_bytes: usize,
) -> Result<Vec<Qwen4ExpertEntry>, Qwen4GpuForwardError> {
    if !matches!(layout, Qwen4ExpertTableLayout::Single) {
        return Err(invalid(
            "qwen4.expert_keep requires a single-device non-tier placement",
        ));
    }
    let mut slot = 0;
    let mut entries = Vec::with_capacity(experts);
    for expert in 0..experts {
        if ids.binary_search(&(expert as u16)).is_err() {
            entries.push(Qwen4ExpertEntry {
                origin: Qwen4ExpertOrigin::Local,
                gate_up_ptr: 0,
                down_ptr: 0,
            });
            continue;
        }
        entries.push(Qwen4ExpertEntry {
            origin: Qwen4ExpertOrigin::Local,
            gate_up_ptr: expert_slot_ptr(
                "pruned gate/up",
                local.gate_up_base,
                local.gate_up_bytes,
                slot,
                gu_bytes,
            )?,
            down_ptr: expert_slot_ptr(
                "pruned down",
                local.down_base,
                local.down_bytes,
                slot,
                dn_bytes,
            )?,
        });
        slot += 1;
    }
    Ok(entries)
}

/// An EP rank's resident tensors hold exactly `per_rank` experts.
fn check_exact_compact(
    local: Qwen4PeerExpertSource,
    per_rank: usize,
    gate_up_expert_bytes: usize,
    down_expert_bytes: usize,
) -> Result<(), Qwen4GpuForwardError> {
    let gate_up = checked_bytes(per_rank, gate_up_expert_bytes)?;
    let down = checked_bytes(per_rank, down_expert_bytes)?;
    if local.gate_up_bytes != gate_up || local.down_bytes != down {
        return Err(invalid(format!(
            "EP compact expert tensors hold {}+{} bytes, expected exactly {per_rank} experts = {gate_up}+{down}",
            local.gate_up_bytes, local.down_bytes
        )));
    }
    Ok(())
}

/// A non-owning byte window over a live peer or tier payload allocation.
/// It is only cut into per-expert views and is never freed here.
fn foreign_packed_source(base: usize, bytes: usize, dtype: DType) -> GpuTensor {
    GpuTensor {
        // SAFETY: `base` names a live device-visible allocation of `bytes`
        // bytes, whose owner is retained by the peer or tier lease, and
        // the wrapper is `Borrowed`, so neither this rank's `free_tensor`
        // nor a drop releases it.
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(base as *mut std::ffi::c_void, bytes) },
        shape: vec![bytes],
        dtype,
    }
}

/// One per-rank zero-filled expert in the routed format (QT44 gate/up, QT53
/// down), shared by every layer's EP decode table for the experts the rank
/// does not own.
///
/// All-zero bytes dequantize to exactly 0, so the producers (which run
/// unchanged on all ten slots) emit zero rows for non-owned slots, which the
/// row gather then overwrites with the owner's bits:
/// * gate/up (`qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip`): the
///   group header is f16 `sc`, f16 `zp` read from bytes `[0,4)` / `[4,8)`;
///   every weight is `sc * nibble + zp` = `0 * 0 + 0` = +0.0 in f32, so each
///   product is `+-0 * x` (x is a finite rotated activation) and every
///   accumulator (initialised to +0.0) stays +0.0;
/// * down (`gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip`):
///   `scale`, `zero` are the f16 at group bytes `[0,2)` / `[2,4)`; the same
///   arithmetic gives `acc = +0.0`, written to the expanded row.
/// No division, reciprocal or NaN/inf source exists in either dequant.
pub(crate) struct Qwen4EpDummyExpert {
    gate_up: GpuTensor,
    down: GpuTensor,
}

impl Qwen4EpDummyExpert {
    /// Encoded bytes of one routed expert `(gate_up, down)`.
    fn expert_bytes(config: &Qwen4Config) -> Result<(usize, usize), Qwen4GpuForwardError> {
        let gate_up = checked_bytes(
            2 * config.moe_intermediate_size,
            row_stride(DType::MQ4G256V2, config.hidden_size),
        )?;
        let down = checked_bytes(
            config.hidden_size,
            row_stride(DType::MQ4G128V2, config.moe_intermediate_size),
        )?;
        if gate_up % 4 != 0 || down % 4 != 0 {
            return Err(invalid("EP dummy expert byte size is not a multiple of 4"));
        }
        Ok((gate_up, down))
    }

    fn new(gpu: &mut Gpu, config: &Qwen4Config) -> Result<Self, Qwen4GpuForwardError> {
        let (gate_up_bytes, down_bytes) = Self::expert_bytes(config)?;
        let gate_up = gpu.zeros(&[gate_up_bytes / 4], DType::F32)?;
        let down = match gpu.zeros(&[down_bytes / 4], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(gate_up);
                return Err(error.into());
            }
        };
        Ok(Self { gate_up, down })
    }

    fn source(&self) -> Qwen4PeerExpertSource {
        Qwen4PeerExpertSource {
            gate_up_base: self.gate_up.buf.as_ptr() as usize,
            gate_up_bytes: self.gate_up.buf.size(),
            down_base: self.down.buf.as_ptr() as usize,
            down_bytes: self.down.buf.size(),
        }
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [self.gate_up, self.down] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// The forward's row gather and gather buffers, lent to the rank's MTP head
/// for one call (buffers `2 + j % 2` carry the head's MoE rows, see
/// [`qwen4_ep_buffer_index`]).
#[derive(Clone, Copy)]
pub(crate) struct MtpEpGather<'a> {
    pub(crate) hook: &'a (dyn MoeRowGather + Send + Sync),
    pub(crate) buffers: [&'a GpuTensor; QWEN4_EP_GATHER_BUFFERS],
}

/// `HIPFIRE_EP_BALANCED_OWNERS` (default on under EP; `=0` disables): per-call
/// balanced expert ownership on the host-mapped trunk MoE layers of an EP rank.
/// Only the exact value `0` is off (unset or any other value is on), and off
/// builds nothing (no tables, no buffer, no stage, no kernel load).
pub const QWEN4_EP_BALANCED_OWNERS_ENV: &str = "HIPFIRE_EP_BALANCED_OWNERS";

fn qwen4_ep_balanced_owners_enabled() -> bool {
    hipfire_config::developer_var(QWEN4_EP_BALANCED_OWNERS_ENV).as_deref() != Ok("0")
}

/// Why an EP rank does not run balanced ownership, or `None` when it does. The
/// per-call owner assignment (`qwen4_ep_owner_assign_pm`) ships for exact
/// gfx1201 and at most [`rdna_compute::moe::QWEN4_EP_OWNER_ASSIGN_MAX_RANKS`]
/// ranks only. Elsewhere (gfx1100, or more ranks) balanced ownership is not
/// admitted: the static `e % N` split is exact, so the rank simply runs without it.
pub(crate) fn qwen4_ep_balanced_owners_refusal(arch: &str, ranks: usize) -> Option<String> {
    let max_ranks = rdna_compute::moe::QWEN4_EP_OWNER_ASSIGN_MAX_RANKS;
    if !rdna_compute::moe::qwen4_ep_owner_assign_pm_arch(arch) {
        Some("needs gfx1201".to_string())
    } else if ranks > max_ranks {
        Some(format!("needs at most {max_ranks} ranks"))
    } else {
        None
    }
}

/// `HIPFIRE_EP_OWNER_ROUTED_PREFILL` (default ON under EP, N = 2..=8; `=0` is the kill switch):
/// owner-routed chunk MoE. A prefill chunk of at least 512 rows runs the grouped GEMMs on the
/// compact (owned-experts-only) table with the tiles of the other ranks' experts masked out, then
/// pulls each other-rank-owned tile's BF16 grouped down rows over P2P (`MoeTileExchange`) instead
/// of computing every expert on every rank. Two routes carry it, both over the stable grouping
/// every rank shares: exact gfx1201 on the symmetric IU4 GEMMs (a chunk whose layers verified
/// symmetric on every rank), and exact gfx1100 on the F16 WMMA grouped GEMMs (gate/up with the
/// SwiGLU epilogue and the down, both BF16-out; no symmetric headers needed, the verdict is
/// that every layer's experts are QT44/QT53). Every other chunk keeps the full-table route.
/// Only the exact value `0` is off (unset or any other value is on, like
/// `HIPFIRE_EP_BALANCED_OWNERS`); every other admission condition still applies, and off
/// allocates nothing, lowers no stage and loads no kernel.
pub const QWEN4_EP_OWNER_ROUTED_PREFILL_ENV: &str = "HIPFIRE_EP_OWNER_ROUTED_PREFILL";

/// The shared EP lever rule: default on, exact `0` off.
fn ep_lever_on(value: Option<&str>) -> bool {
    value != Some("0")
}

fn qwen4_ep_owner_routed_prefill_enabled() -> bool {
    ep_lever_on(
        hipfire_config::developer_var(QWEN4_EP_OWNER_ROUTED_PREFILL_ENV)
            .ok()
            .as_deref(),
    )
}

/// Rank counts the owner-routed chunk MoE admits: EP2 through EP8, matching the mesh
/// (`attach_ep_mesh`) and the eight-source pull launcher ABI.
const fn tile_ranks_supported(ranks: usize) -> bool {
    matches!(ranks, 2..=8)
}

/// The little-endian i32 owner of every expert the launcher reads, from the
/// sealed ownership ([`crate::ep_seal::Qwen4EpSeal::owner_ranks`]).
fn tile_owner_table_bytes(owners: &[usize]) -> Vec<u8> {
    owners
        .iter()
        .flat_map(|&owner| (owner as i32).to_le_bytes())
        .collect()
}

/// Per-rank donor mirror bytes of one G2 stage layer (`[gate/up, down]`): rank `rank` of
/// `ranks` hosts `Qwen4EpPlacement::local_count(experts)` experts of each kind, which covers
/// slot `e / ranks` of every expert it owns.
fn stage_donor_bytes(experts: usize, rank: usize, ranks: usize) -> [usize; 2] {
    let local_experts = Qwen4EpPlacement { rank, ranks }.local_count(experts);
    [
        local_experts * QWEN4_EXPERT_STAGE_GATE_UP_BYTES,
        local_experts * QWEN4_EXPERT_STAGE_DOWN_BYTES,
    ]
}

/// Redirect the entries of the experts this rank owns (`e % ranks == rank`) into its donor
/// buffer at `base`, slot `e / ranks`; every other entry keeps its pointer.
fn redirect_owned_entries(
    placement: Qwen4EpPlacement,
    base: usize,
    stride: usize,
    entries: &[usize],
) -> Vec<u8> {
    entries
        .iter()
        .enumerate()
        .flat_map(|(expert, &pointer)| {
            let pointer = if expert % placement.ranks == placement.rank {
                base + (expert / placement.ranks) * stride
            } else {
                pointer
            };
            pointer.to_ne_bytes()
        })
        .collect()
}

/// One rank's owner-routed chunk MoE state. The buffers exist from forward construction
/// (before the mesh enables peers, so a peer-mapped allocation never lags the mapping);
/// the hook is installed once every rank agreed on the per-layer verdicts (symmetric IU4
/// headers on gfx1201, QT44/QT53 formats for the F16 WMMA route on gfx1100).
pub(crate) struct Qwen4TileRoute {
    /// Owner rank of every expert (`i32` per expert, `e % ranks`): the mask and the pull
    /// read it as an input.
    owner_table: GpuTensor,
    /// The masked tile ids gate/up and down read (`i32` per tile of the padded capacity).
    tile_ids_owned: GpuTensor,
    /// Padded tile capacity the exchange is registered for.
    max_tiles: usize,
    /// Per layer: the compact layer's verdict, taken on the full tables in `attach_ep`
    /// (gfx1201: the symmetric IU4 header verdict; gfx1100: the expert formats the F16 WMMA
    /// grouped GEMMs take). Empty until then.
    verdicts: Vec<bool>,
    /// The route is the F16 WMMA grouped route (exact gfx1100) rather than the symmetric IU4
    /// one (exact gfx1201): the compact layers keep their plain grouped declaration and the
    /// chunk gate asks the WMMA predicate.
    wmma: bool,
    /// The installed transport (`None` until the mesh agreed).
    hook: Option<Arc<dyn MoeTileExchange + Send + Sync>>,
}

impl Qwen4TileRoute {
    /// Admit the lever for this load or log why not and return `None` (nothing allocated).
    #[allow(clippy::too_many_arguments)]
    fn admit(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        owners: &[usize],
        scratch: &Qwen4GpuForwardScratch,
        mode: Qwen4ForwardScratchMode,
        max_chunk: usize,
        flash_next: bool,
        exchange: Option<&ep_reserve_scratch::EpExchangeCapacity>,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let refuse = |reason: &str| {
            eprintln!(
                "qwen4 EP rank {}/{}: owner-routed prefill not admitted ({reason}); chunks keep the full-table route",
                placement.rank, placement.ranks
            );
            Ok(None)
        };
        if !tile_ranks_supported(placement.ranks) {
            return refuse("needs 2..=8 ranks");
        }
        if !rdna_compute::moe::ep_tile_pm_arch(&gpu.arch) {
            return refuse("needs gfx1100 or gfx1201");
        }
        if !flash_next {
            return refuse("needs the Flash-Next profile");
        }
        // gfx1201 runs the symmetric IU4 route; gfx1100 the F16 WMMA grouped route (BF16-out
        // gate/up with SwiGLU and BF16-out down), which needs no symmetric headers.
        let wmma = gpu.arch == "gfx1100";
        if wmma {
            if !gpu.flags.moe_grouped_gemm
                || !gpu.qwen4_moe_f16_wmma_grouped_applies(
                    rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS,
                    config.moe_intermediate_size,
                    config.hidden_size,
                )
            {
                return refuse("the F16 WMMA grouped MoE route is not available");
            }
        } else if !gpu.flags.moe_grouped_gemm || !gpu.qwen4_moe_sym_iu4_requested() {
            return refuse("the symmetric IU4 route is not requested");
        }
        if mode != Qwen4ForwardScratchMode::Full {
            // The compact plan lends `y_down_grouped` to a phase-disjoint field the peer
            // would be reading.
            return refuse("the compact scratch plan aliases y_down_grouped");
        }
        // Grouping scratch (gfx1100) or the full sym scratch (gfx1201) before the mesh enables
        // peers (the grouped-row buffer and the masked ids are allocated now; ROCm does not
        // map later allocations).
        if wmma {
            gpu.reserve_qwen4_moe_group_scratch(
                max_chunk,
                config.num_experts_per_tok,
                config.num_experts,
            )?;
        } else {
            gpu.reserve_qwen4_moe_sym(
                max_chunk,
                config.num_experts_per_tok,
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )?;
        }
        let word = std::mem::size_of::<i32>();
        let tile_bytes = 16 * config.hidden_size * DType::BF16.size();
        let (tile_capacity, max_tiles) = match exchange {
            // EP reserve: the registered capacity is the 4096-row prompt plan's
            // (persistent grouped-down root and masked tile ids), never the
            // selected resident bank's.
            Some(capacity) => (
                capacity.tile_ids,
                ep_reserve_forward::exchange_max_tiles(capacity, config.hidden_size),
            ),
            None => {
                let tile_capacity = scratch.moe_expert_tile_ids.buf.size() / word;
                (
                    tile_capacity,
                    tile_capacity.min(scratch.moe_y_down_grouped.buf.size() / tile_bytes),
                )
            }
        };
        if max_tiles == 0 {
            return refuse("no tile capacity");
        }
        if owners.len() != config.num_experts {
            return Err(invalid(
                "the sealed owner table does not cover every expert",
            ));
        }
        let owners = tile_owner_table_bytes(owners);
        let owner_table = gpu.upload_raw(&owners, &[owners.len()])?;
        let sentinels = vec![0xFFu8; tile_capacity * word];
        let tile_ids_owned = match gpu.upload_raw(&sentinels, &[sentinels.len()]) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(owner_table);
                return Err(error.into());
            }
        };
        Ok(Some(Self {
            owner_table,
            tile_ids_owned,
            max_tiles,
            verdicts: Vec::new(),
            wmma,
            hook: None,
        }))
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [self.owner_table, self.tile_ids_owned] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// One host-mapped trunk MoE layer under balanced ownership: the rank's *live*
/// expert runtime. It is the layer's full (P2P) addressing
/// ([`Qwen4MoeLayerRuntime::from_moe_full`]); `qwen4_ep_owner_assign` rewrites
/// its two pointer tables every call (owned routed experts keep their full
/// address, every other entry names the rank's zero dummy expert), and the
/// routed producers read those tables in place of the static EP decode table.
pub(crate) struct Qwen4BalancedLayer {
    live: Qwen4MoeLayerRuntime,
    /// The layer's label (`layer.layer`, what the dispatch hook is called with).
    label: usize,
    tables: Qwen4EpOwnerTables,
}

/// What the row-gather hook needs of a rank's balanced layers: the owner-assign
/// table addresses per layer label and a non-owning view of the rank's
/// `slot_owner` buffer (owned by [`Qwen4ForwardEp`], freed with it).
pub(crate) struct Qwen4BalancedPlan {
    pub(crate) layers: Vec<(usize, Qwen4EpOwnerTables)>,
    pub(crate) slot_owner: GpuTensor,
}

/// Expert-parallel state of one rank's forward (present iff the bundle's
/// weights carry an EP placement).
pub(crate) struct Qwen4ForwardEp {
    placement: Qwen4EpPlacement,
    /// The rank's binding to its load's seal: every EP dispatch table and the
    /// owner tables derive from it.
    seal: Qwen4EpRankSeal,
    dummy: Qwen4EpDummyExpert,
    /// Per-layer full (P2P) tables for chunk forwards; empty until
    /// [`Qwen4GpuForward::attach_ep`].
    moe_full: Vec<Qwen4MoeLayerRuntime>,
    /// Expanded-row exchange buffers, see [`qwen4_ep_buffer_index`].
    buffers: [GpuTensor; QWEN4_EP_GATHER_BUFFERS],
    /// Row gather the one-row and few-row forwards install, once attached.
    gather: Option<Arc<dyn MoeRowGather + Send + Sync>>,
    /// Balanced-ownership state per MoE ordinal (`Some` for every host-mapped
    /// trunk layer); empty when `HIPFIRE_EP_BALANCED_OWNERS=0` at
    /// [`Qwen4GpuForward::attach_ep`].
    balanced: Vec<Option<Qwen4BalancedLayer>>,
    /// The rank's `QWEN4_EP_GATHER_MAX_ROWS * k` i32 slot owners, shared by the
    /// balanced layers (stream-ordered: a call's assignment writes it and the
    /// same call's gather reads it before the next layer's assignment).
    slot_owner: Option<GpuTensor>,
    /// Owner-routed chunk MoE state (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`, admitted at
    /// construction); `None` when the lever is off or refused.
    tile: Option<Qwen4TileRoute>,
}

impl Qwen4ForwardEp {
    fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        seal: Qwen4EpRankSeal,
    ) -> Result<Self, Qwen4GpuForwardError> {
        validate_ep_placement(placement, config.num_experts)?;
        let elements = QWEN4_EP_GATHER_MAX_ROWS
            .checked_mul(config.num_experts_per_tok)
            .and_then(|slots| slots.checked_mul(config.hidden_size))
            .ok_or_else(|| invalid("EP gather buffer size overflows"))?;
        let dummy = Qwen4EpDummyExpert::new(gpu, config)?;
        let mut buffers = Vec::with_capacity(QWEN4_EP_GATHER_BUFFERS);
        for _ in 0..QWEN4_EP_GATHER_BUFFERS {
            match gpu.zeros(&[elements], DType::F32) {
                Ok(tensor) => buffers.push(tensor),
                Err(error) => {
                    for tensor in buffers {
                        let _ = gpu.free_tensor(tensor);
                    }
                    let _ = dummy.free_gpu(gpu);
                    return Err(error.into());
                }
            }
        }
        let buffers: [GpuTensor; QWEN4_EP_GATHER_BUFFERS] = match buffers.try_into() {
            Ok(buffers) => buffers,
            Err(_) => return Err(invalid("EP gather buffer count")),
        };
        let state = Self {
            placement,
            seal,
            dummy,
            moe_full: Vec::new(),
            buffers,
            gather: None,
            balanced: Vec::new(),
            slot_owner: None,
            tile: None,
        };
        // The zero fills are stream-ordered; peers read the buffers over P2P
        // on their own streams.
        if let Err(error) = gpu.hip.device_synchronize() {
            let _ = state.free_gpu(gpu);
            return Err(error.into());
        }
        Ok(state)
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for layer in self.moe_full {
            if let Some(error) = layer.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        for layer in self.balanced.into_iter().flatten() {
            if let Some(error) = layer.live.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some(route) = self.tile {
            if let Some(error) = route.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some(tensor) = self.slot_owner {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        for tensor in self.buffers {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        if let Some(error) = self.dummy.free_gpu(gpu) {
            first.get_or_insert(error);
        }
        first
    }
}

/// The stock-FP8 layer's two qt=55 expert tables, validated against the
/// config's fixed expert geometry (`None` for every other recipe).  The
/// tables come from the loader accessor, which already proved each is one
/// fully resident, 256-byte-aligned `[E, rows, K]` buffer; only the
/// single-device layout exists for them (the recipe refuses expert
/// parallelism at load).
fn fp8_expert_tables(
    weights: &Qwen4Weights,
    moe: &MoeWeights,
    config: &Qwen4Config,
    layout: &Qwen4ExpertTableLayout<'_>,
    layer_label: usize,
) -> Result<Option<Fp8MoeExpertTables>, Qwen4GpuForwardError> {
    if !weights.manifest.identity.recipe.is_stock_fp8() {
        return Ok(None);
    }
    if !matches!(layout, Qwen4ExpertTableLayout::Single) {
        return Err(invalid(format!(
            "layer {layer_label}: qt=55 FP8 experts bind only the single-device table"
        )));
    }
    let tables = weights.fp8_moe_expert_tables(moe)?;
    let (gate_up, down) = (tables.gate_up, tables.down);
    if (gate_up.experts, gate_up.rows, gate_up.k)
        != (
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        || (down.experts, down.rows, down.k)
            != (
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
    {
        return Err(invalid(format!(
            "layer {layer_label}: qt=55 expert tables [{}, {}, {}] / [{}, {}, {}] disagree with the \
             config's [{}, {}, {}] / [{}, {}, {}]",
            gate_up.experts,
            gate_up.rows,
            gate_up.k,
            down.experts,
            down.rows,
            down.k,
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )));
    }
    Ok(Some(tables))
}

/// The gfx942 qt=55 grouped route runs in the `Full` scratch plan's F32
/// buffers, preallocated at construction (no per-call allocation, no
/// dependence on WMMA): `G = grouped_rows` is the padded-16 slot bound of
/// `max_chunk` rows, so the grouped gate_up output is `[G, 2*moe]` F32, the
/// grouped down output `[G, hidden]` F32, `rot_batch` holds `slots * moe`
/// F32, and the tile descriptor lives in `moe_gate_batch` (16 bytes per
/// 16-row tile, `G` bytes). Refuses a plan that would not hold them (the
/// compact gfx1201 plan, or a changed layout) before the first launch.
fn check_fp8_grouped_scratch(
    scratch: &Qwen4GpuForwardScratch,
    config: &Qwen4Config,
    max_chunk: usize,
) -> Result<(), Qwen4GpuForwardError> {
    if scratch.mode != Qwen4ForwardScratchMode::Full {
        return Err(invalid("the qt=55 FP8 route needs the Full scratch plan"));
    }
    let layout = Qwen4ScratchLayout::for_rows(program_dims(config), max_chunk)
        .map_err(|error| invalid(error.to_string()))?;
    let f32_bytes = DType::F32.size();
    let descriptor_bytes = layout.grouped_rows / MOE_GROUPED_BLOCK_M * 16;
    for (name, tensor, need) in [
        (
            "grouped gate_up output",
            &scratch.moe_y_gate_up_grouped,
            scratch_mul(
                scratch_mul(layout.grouped_rows, 2 * config.moe_intermediate_size)?,
                f32_bytes,
            )?,
        ),
        (
            "grouped down output",
            &scratch.moe_y_down_grouped,
            scratch_mul(
                scratch_mul(layout.grouped_rows, config.hidden_size)?,
                f32_bytes,
            )?,
        ),
        (
            "rot_batch",
            &scratch.moe_rot_batch,
            scratch_mul(layout.routed_rows, f32_bytes)?,
        ),
        (
            "tile descriptor (gate_batch)",
            &scratch.moe_gate_batch,
            descriptor_bytes,
        ),
        (
            "sorted slot index",
            &scratch.moe_sorted_slot_index,
            scratch_mul(layout.grouped_rows, std::mem::size_of::<i32>())?,
        ),
    ] {
        if tensor.buf.size() < need {
            return Err(invalid(format!(
                "qt=55 FP8 {name} scratch holds {} bytes, {max_chunk} rows need {need}",
                tensor.buf.size()
            )));
        }
    }
    for (name, tensor) in [
        ("grouped gate_up output", &scratch.moe_y_gate_up_grouped),
        ("grouped down output", &scratch.moe_y_down_grouped),
    ] {
        if tensor.dtype != DType::F32 {
            return Err(invalid(format!(
                "qt=55 FP8 {name} must be F32, not {:?}",
                tensor.dtype
            )));
        }
    }
    Ok(())
}

impl Qwen4MoeLayerRuntime {
    /// The route this layer declares for a `rows`-row (routing rows) call: a
    /// stock-FP8 layer is always [`MoeRouteCapability::Fp8B128Grouped`]; a
    /// QT44/QT53 layer is the symmetric variant only once verified and the
    /// chunk reaches the IU4 arm's row floor.
    pub(crate) fn route_capability(&self, rows: usize) -> MoeRouteCapability {
        if self.slab.is_some() {
            MoeRouteCapability::SlabCodecGrouped
        } else if self.fp8 {
            MoeRouteCapability::Fp8B128Grouped
        } else if self.symmetric && rows >= rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS {
            MoeRouteCapability::Qt44Qt53GroupedSymmetric
        } else {
            MoeRouteCapability::Qt44Qt53Grouped
        }
    }

    /// The sealed expert-slab layer's cold codec for its route policy
    /// ([`MoeRoutePolicy::slab_cold_codec`]): W3 (1) or E8P (3), `None` for an
    /// MQ4-only slab layer and for every non-slab layer. Read from the layer's
    /// load-time metadata, never from a table scan.
    pub(crate) fn slab_cold_codec(&self) -> Option<u8> {
        self.slab
            .as_ref()
            .map(|slab| slab.cold_codec)
            .filter(|codec| *codec != crate::expert_arena::CODEC_MQ4)
    }

    /// The BorrowPrefill runtime of a decode-share load for prefill capacity
    /// class `class`: the same layer built from the lease's class view (the
    /// donors that class can overwrite at their pinned host mirror, the rest
    /// at their device bytes). Serves only prompt tiles of at least the
    /// resident bank's rows; never arms a CPU splice or D2 split.
    pub(crate) fn from_moe_borrow(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        d2_default: bool,
        class: usize,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Single,
            None,
            None,
            true,
            d2_default,
            crate::expert_tier_owners::ExpertReadView::BorrowPrefill(class),
            false,
            None,
        )
    }

    /// Single-device runtime: the layer's resident tensors hold every expert.
    pub(crate) fn from_moe(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        d2_default: bool,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Single,
            None,
            None,
            true,
            d2_default,
            crate::expert_tier_owners::ExpertReadView::Resident,
            false,
            None,
        )
    }

    /// MTP head labels are not trunk indices; its named payload stays untiered.
    pub(crate) fn from_moe_untiered(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        head_split: bool,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Single,
            None,
            None,
            false,
            false,
            crate::expert_tier_owners::ExpertReadView::Resident,
            head_split,
            None,
        )
    }

    pub(crate) fn tier_lease(&self) -> Option<&crate::expert_tier_owners::ExpertTierLease> {
        self.tier.as_ref()
    }

    /// Actual per-layer D2 split executor storage for the bundle census: the
    /// live executor when this layer is armed, `None` otherwise. Pure (no
    /// allocation, mutation or qualification); capacity and qualified reads
    /// stay on their own getters.
    pub(crate) fn cpu_split_storage_executor(
        &self,
    ) -> Option<&hipfire_dispatch::cpu_split::CpuSplitExecutor> {
        self.cpu_split.as_ref()
    }

    /// EP decode runtime: the resident expert tensors are this rank's compact
    /// `placement.local_count(num_experts)` experts; owned experts name them, every other
    /// expert names `dummy`. Dispatch binds the sealed plan of expert group `group`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_moe_ep(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        dummy: &Qwen4EpDummyExpert,
        seal: &Qwen4EpRankSeal,
        group: usize,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let sealed = seal
            .expert_binding(group, placement, gpu.device_id)
            .map_err(invalid)?;
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy.source(),
            },
            Some(dummy),
            Some(sealed),
            true,
            false,
            crate::expert_tier_owners::ExpertReadView::Resident,
            false,
            None,
        )
    }

    /// EP reserve borrowed view: the compact runtime of a scratch-donor layer whose
    /// device bytes are scratch while a borrowed epoch is open. Owned experts name
    /// the loan's pinned mirror (`gate_up` / `down`: typed host-mapped descriptors of
    /// the loan-owned allocation), every other expert the zero dummy, exactly as
    /// [`Self::from_moe_ep`] does over the device tensors (same sealed binding of
    /// expert group `group`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_moe_ep_mirror(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        dummy: &Qwen4EpDummyExpert,
        seal: &Qwen4EpRankSeal,
        group: usize,
        gate_up: &GpuTensor,
        down: &GpuTensor,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let sealed = seal
            .expert_binding(group, placement, gpu.device_id)
            .map_err(invalid)?;
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy.source(),
            },
            Some(dummy),
            Some(sealed),
            true,
            false,
            crate::expert_tier_owners::ExpertReadView::Resident,
            false,
            Some((gate_up, down)),
        )
    }

    /// EP chunk runtime: owned experts name the local compact tensors, every
    /// other expert names the owning peer's compact tensors
    /// (`peers_for_layer[rank]`). Dispatch binds the sealed plan of `layer_label`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_moe_full(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        placement: Qwen4EpPlacement,
        peers_for_layer: &[Qwen4PeerExpertSource],
        seal: &Qwen4EpRankSeal,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let sealed = seal
            .expert_binding(layer_label, placement, gpu.device_id)
            .map_err(invalid)?;
        Self::build(
            gpu,
            weights,
            moe,
            layer_label,
            config,
            Qwen4ExpertTableLayout::Full {
                placement,
                peers: peers_for_layer,
            },
            None,
            Some(sealed),
            true,
            false,
            crate::expert_tier_owners::ExpertReadView::Resident,
            false,
            None,
        )
    }

    /// Single-device tables are one rank-0-of-1 dispatch table. EP tables
    /// (`sealed`, required exactly for the `Ep` and `Full` layouts) are the
    /// rank's binding adapted from the sealed plan: real owners and compact
    /// slots, while the views name whatever allocation each entry uses
    /// (local, zero dummy or peer).
    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
        layout: Qwen4ExpertTableLayout<'_>,
        dummy: Option<&Qwen4EpDummyExpert>,
        sealed: Option<(ExpertTable, ExpertBindingCache)>,
        allow_tiers: bool,
        d2_default: bool,
        read_view: crate::expert_tier_owners::ExpertReadView,
        head_split: bool,
        source_override: Option<(&GpuTensor, &GpuTensor)>,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let ep_layout = matches!(
            layout,
            Qwen4ExpertTableLayout::Ep { .. } | Qwen4ExpertTableLayout::Full { .. }
        );
        if ep_layout != sealed.is_some() {
            return Err(invalid(format!(
                "layer {layer_label}: EP tables need the sealed rank binding, single-device tables none"
            )));
        }
        let router_source = weights.resident(&moe.gate)?;
        let scalar_source = weights.resident(&moe.shared_gate_scalar)?;
        let shared_gate_source = weights.resident(&moe.shared_gate)?;
        let shared_up_source = weights.resident(&moe.shared_up)?;
        let shared_down_source = weights.resident(&moe.shared_down)?;
        let router =
            ProjectionView::from_source(router_source, config.num_experts, config.hidden_size);
        let shared_scalar = ProjectionView::from_source(scalar_source, 1, config.hidden_size);
        let shared_gate = ProjectionView::from_source(
            shared_gate_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_up = ProjectionView::from_source(
            shared_up_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_down = ProjectionView::from_source(
            shared_down_source,
            config.hidden_size,
            config.shared_expert_intermediate_size,
        );

        let tier_layer = if allow_tiers {
            weights.expert_tier_layer_view(layer_label, read_view)
        } else {
            None
        };
        let borrow_view = matches!(
            read_view,
            crate::expert_tier_owners::ExpertReadView::BorrowPrefill(_)
        );
        if borrow_view && tier_layer.is_none() {
            return Err(invalid(format!(
                "layer {layer_label}: no BorrowPrefill expert view (the load carries no decode-share \
                 binding for this layer)"
            )));
        }
        if tier_layer.is_some()
            && (!matches!(layout, Qwen4ExpertTableLayout::Single)
                || moe.expert_arena.is_some()
                || weights.manifest.identity.recipe.is_stock_fp8())
        {
            return Err(invalid(format!(
                "layer {layer_label}: expert tiers require the single-device aggregate MQ4 route (no EP, slab arena or stock FP8)"
            )));
        }
        let layout = match tier_layer.as_ref() {
            Some(view) => Qwen4ExpertTableLayout::Tier {
                gate_up: view.gate_up_ptrs(),
                down: view.down_ptrs(),
            },
            None => layout,
        };
        if let Some(arena_ref) = moe.expert_arena.as_ref() {
            // A slab layer never takes the D1 splice, so the explicit `cpu`
            // mode requests its D2 split exactly as it does a tier layer's.
            let split_requested = (hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled()
                && !hipfire_dispatch::cpu_exec::passback_enabled()
                || d2_default)
                && allow_tiers;
            return Self::build_slab(
                gpu,
                weights,
                arena_ref,
                layer_label,
                config,
                &layout,
                [router, shared_scalar, shared_gate, shared_up, shared_down],
                split_requested,
            );
        }
        // Tier loads deliberately have no resident aggregate expert tensors.
        // Whole-owner descriptors supply the fixed MQ4 dtype only. Tier
        // entries below always use their own checked absolute addresses.
        let tier_source_range = tier_layer
            .as_ref()
            .map(|view| {
                use crate::expert_tier_plan::Placement;
                view.owner_range(Placement::Device)
                    .or_else(|| view.owner_range(Placement::HostMapped))
                    .ok_or_else(|| invalid("expert tiers have no payload owner"))
            })
            .transpose()?;
        let tier_gate_source = tier_source_range
            .map(|(base, bytes)| foreign_packed_source(base, bytes, DType::MQ4G256V2));
        let tier_down_source = tier_source_range
            .map(|(base, bytes)| foreign_packed_source(base, bytes, DType::MQ4G128V2));
        let gate_up_source = match tier_gate_source.as_ref() {
            Some(source) => source,
            None => match source_override {
                Some((gate_up, _)) => gate_up,
                None => weights.resident(&moe.experts_gate_up)?,
            },
        };
        let down_source = match tier_down_source.as_ref() {
            Some(source) => source,
            None => match source_override {
                Some((_, down)) => down,
                None => weights.resident(&moe.experts_down)?,
            },
        };
        let gate_up_dtype = gate_up_source.dtype;
        let down_dtype = down_source.dtype;
        let gate_up_rows = 2 * config.moe_intermediate_size;
        let gate_up_row_stride = row_stride(gate_up_dtype, config.hidden_size);
        let down_row_stride = row_stride(down_dtype, config.moe_intermediate_size);
        // Stock FP8: the loader accessor's immutable qt=55 tables. Expert `e`
        // is the block at `base + e * stride` (codes, then its BF16 `s_inv`
        // grid), so an expert's byte extent is the block stride, not rows *
        // row_stride; the view's `row_stride` is the code row (`k`) and the
        // block's pad and scale grid ride in its encoded bytes.
        let fp8_tables = if tier_layer.is_some() {
            None
        } else {
            fp8_expert_tables(weights, moe, config, &layout, layer_label)?
        };
        let (gate_up_expert_bytes, down_expert_bytes) = match &fp8_tables {
            Some(tables) => (tables.gate_up.stride, tables.down.stride),
            None => (
                checked_bytes(gate_up_rows, gate_up_row_stride)?,
                checked_bytes(config.hidden_size, down_row_stride)?,
            ),
        };
        let resident_format = if fp8_tables.is_some() {
            "qwen4-resident-fp8b128-v1"
        } else {
            "qwen4-resident-mqv2-v1"
        };
        let kept_ids = config.kept_expert_ids(layer_label, moe.gate.name.starts_with("mtp."));
        let source_experts = if let Some(ids) = kept_ids {
            match &layout {
                Qwen4ExpertTableLayout::Ep { placement, .. }
                | Qwen4ExpertTableLayout::Full { placement, .. } => {
                    ids.iter().filter(|&&e| placement.owns(e as usize)).count()
                }
                _ => ids.len(),
            }
        } else {
            layout.local_experts(config.num_experts)?
        };
        let expected_gate_up_bytes = checked_bytes(source_experts, gate_up_expert_bytes)?;
        let expected_down_bytes = checked_bytes(source_experts, down_expert_bytes)?;
        if tier_layer.is_none() && gate_up_source.buf.size() < expected_gate_up_bytes {
            return Err(invalid(format!(
                "layer {} gate/up packed bytes {} < expected {}",
                layer_label,
                gate_up_source.buf.size(),
                expected_gate_up_bytes
            )));
        }
        if tier_layer.is_none() && down_source.buf.size() < expected_down_bytes {
            return Err(invalid(format!(
                "layer {} down packed bytes {} < expected {}",
                layer_label,
                down_source.buf.size(),
                expected_down_bytes
            )));
        }
        if let Some(view) = tier_layer.as_ref() {
            if config.num_experts != view.gate_up_ptrs().len()
                || config.num_experts != view.down_ptrs().len()
                || config.num_experts != view.codec_table().len()
                || config.hidden_size != 2560
                || config.moe_intermediate_size != 640
            {
                return Err(invalid("expert tiers require Flash-Next MQ4 geometry"));
            }
            for eid in 0..config.num_experts {
                let entry = view
                    .entry(eid)
                    .ok_or_else(|| invalid("missing tier expert entry"))?;
                if entry.format != crate::expert_slabs::ExpertFormat::Mq4Xts {
                    return Err(invalid("aggregate expert tiers require MQ4 entries"));
                }
                let owner = view.owner_range(entry.tier);
                tier_expert_extent(
                    "gate/up",
                    owner,
                    entry.gate_up_offset,
                    entry.gate_up_len,
                    gate_up_expert_bytes,
                    view.gate_up_ptrs()[eid],
                )?;
                tier_expert_extent(
                    "down",
                    owner,
                    entry.down_offset,
                    entry.down_len,
                    down_expert_bytes,
                    view.down_ptrs()[eid],
                )?;
            }
        }
        // EP: owned experts keep this rank's residency. A host-mapped compact
        // tensor is device-visible exactly as in the single route, a non-owned
        // expert names the device dummy (decode) or the owner's compact tensor
        // (chunk). Peer pointers into pinned host memory rely on ROCm's UVA:
        // pinned host allocations are device-visible from every GPU, so a
        // peer's host-mapped compact tensor is read through the same address.
        let local_host_mapped = tier_layer.as_ref().map_or_else(
            || gate_up_source.buf.is_host_mapped(),
            |view| view.any_host(),
        );
        if matches!(layout, Qwen4ExpertTableLayout::Ep { .. })
            && (gate_up_dtype != DType::MQ4G256V2 || down_dtype != DType::MQ4G128V2)
        {
            return Err(invalid(format!(
                "layer {layer_label} EP dummy expert is QT44/QT53 but the experts are {gate_up_dtype:?}/{down_dtype:?}"
            )));
        }

        let (gate_up_source_name, down_source_name) = match &moe.expert_source_layout {
            ExpertSourceLayout::PackedFused { gate_up, down, .. } => {
                (gate_up.as_str(), down.as_str())
            }
            layout => {
                return Err(invalid(format!(
                    "layer {} uses unsupported expert source layout {layout:?}",
                    layer_label
                )));
            }
        };
        let local = Qwen4PeerExpertSource {
            gate_up_base: gate_up_source.buf.as_ptr() as usize,
            gate_up_bytes: gate_up_source.buf.size(),
            down_base: down_source.buf.as_ptr() as usize,
            down_bytes: down_source.buf.size(),
        };
        let table_entries: Vec<Qwen4ExpertEntry> = match &fp8_tables {
            _ if kept_ids.is_some() => pruned_expert_entries(
                &layout,
                local,
                config.num_experts,
                kept_ids.unwrap(),
                gate_up_expert_bytes,
                down_expert_bytes,
            )?,
            // The accessor's per-expert block addresses, in expert order: the
            // same pointers the grouped launch reads and the views must name.
            Some(tables) => tables
                .gate_up
                .block_entries()
                .into_iter()
                .zip(tables.down.block_entries())
                .map(|(gate_up_ptr, down_ptr)| Qwen4ExpertEntry {
                    origin: Qwen4ExpertOrigin::Local,
                    gate_up_ptr,
                    down_ptr,
                })
                .collect(),
            None => qwen4_expert_table_entries(
                &layout,
                local,
                config.num_experts,
                gate_up_expert_bytes,
                down_expert_bytes,
            )
            .map_err(|error| match error {
                Qwen4GpuForwardError::Invalid(message) => {
                    invalid(format!("layer {layer_label}: {message}"))
                }
                other => other,
            })?,
        };
        // Whole-tensor windows over each peer rank's compact experts; the
        // per-expert views below are cut from them. `qwen4_expert_table_entries`
        // proved every slot lies inside its window.
        let peer_sources: Vec<(GpuTensor, GpuTensor)> = match layout {
            Qwen4ExpertTableLayout::Full { peers, .. } => peers
                .iter()
                .map(|peer| {
                    (
                        foreign_packed_source(peer.gate_up_base, peer.gate_up_bytes, gate_up_dtype),
                        foreign_packed_source(peer.down_base, peer.down_bytes, down_dtype),
                    )
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut experts = Vec::with_capacity(config.num_experts);
        let mut metadata = Vec::with_capacity(config.num_experts);
        for (expert, entry) in table_entries.iter().enumerate() {
            let absent =
                expert_pruned(kept_ids, expert) && entry.gate_up_ptr == 0 && entry.down_ptr == 0;
            let tier_gate = matches!(entry.origin, Qwen4ExpertOrigin::Tier).then(|| {
                foreign_packed_source(entry.gate_up_ptr, gate_up_expert_bytes, gate_up_dtype)
            });
            let tier_down = matches!(entry.origin, Qwen4ExpertOrigin::Tier)
                .then(|| foreign_packed_source(entry.down_ptr, down_expert_bytes, down_dtype));
            let (gate_up_backing, gate_up_offset, down_backing, down_offset) = match entry.origin {
                Qwen4ExpertOrigin::Local => (
                    gate_up_source,
                    if absent {
                        0
                    } else {
                        entry.gate_up_ptr - local.gate_up_base
                    },
                    down_source,
                    if absent {
                        0
                    } else {
                        entry.down_ptr - local.down_base
                    },
                ),
                Qwen4ExpertOrigin::Tier => (
                    tier_gate.as_ref().expect("tier gate descriptor"),
                    0,
                    tier_down.as_ref().expect("tier down descriptor"),
                    0,
                ),
                Qwen4ExpertOrigin::Dummy => {
                    let dummy = dummy.ok_or_else(|| {
                        invalid(format!(
                            "layer {layer_label}: EP table without a dummy expert"
                        ))
                    })?;
                    (&dummy.gate_up, 0, &dummy.down, 0)
                }
                Qwen4ExpertOrigin::Peer(rank) => {
                    let (peer_gate_up, peer_down) = peer_sources.get(rank).ok_or_else(|| {
                        invalid(format!(
                            "layer {layer_label}: peer rank {rank} has no source"
                        ))
                    })?;
                    (
                        peer_gate_up,
                        entry.gate_up_ptr - peer_gate_up.buf.as_ptr() as usize,
                        peer_down,
                        entry.down_ptr - peer_down.buf.as_ptr() as usize,
                    )
                }
            };
            let mut gate_up = if absent {
                ProjectionView::absent(gate_up_dtype, gate_up_rows, config.hidden_size)
            } else {
                ProjectionView::from_packed(
                    gate_up_backing,
                    gate_up_offset,
                    gate_up_expert_bytes,
                    gate_up_dtype,
                    gate_up_rows,
                    config.hidden_size,
                )
            };
            let mut down = if absent {
                ProjectionView::absent(down_dtype, config.hidden_size, config.moe_intermediate_size)
            } else {
                ProjectionView::from_packed(
                    down_backing,
                    down_offset,
                    down_expert_bytes,
                    down_dtype,
                    config.hidden_size,
                    config.moe_intermediate_size,
                )
            };
            if let Some(view) = tier_layer.as_ref() {
                let host =
                    view.placement(expert) == Some(crate::expert_tier_plan::Placement::HostMapped);
                gate_up.host_mapped = host;
                down.host_mapped = host;
            }
            if gate_up.tensor.buf.as_ptr() as usize != entry.gate_up_ptr
                || down.tensor.buf.as_ptr() as usize != entry.down_ptr
            {
                return Err(invalid(format!(
                    "layer {layer_label} expert {expert} view disagrees with its table entry"
                )));
            }
            experts.push(Qwen4ExpertView { gate_up, down });

            let gate_up_resource = ExpertResource::new(
                gate_up_source_name,
                resident_format,
                vec![gate_up_rows, config.hidden_size],
                gate_up_dtype,
                gate_up_expert_bytes,
                gate_up_row_stride,
                16,
                dtype_rotation_plan(gate_up_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let down_resource = ExpertResource::new(
                down_source_name,
                resident_format,
                vec![config.hidden_size, config.moe_intermediate_size],
                down_dtype,
                down_expert_bytes,
                down_row_stride,
                16,
                dtype_rotation_plan(down_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let resources =
                ExpertResources::fused(gate_up_resource, down_resource).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert resources: {error:?}"))
                })?;
            let record = ExpertMetadata::new(expert, 0, expert, resources).map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert metadata: {error:?}"))
            })?;
            metadata.push(if absent { record.into_absent() } else { record });
        }
        let (table, mut cache) = match sealed {
            Some(sealed) => sealed,
            None => {
                let table = ExpertTable::new(metadata).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert table: {error:?}"))
                })?;
                let cache = table
                    .prepare_binding(0, 1, gpu.device_id)
                    .map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!("expert binding: {error:?}"))
                    })?;
                (table, cache)
            }
        };

        let gate_entries = experts
            .iter()
            .map(|expert| expert.gate_up.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let down_entries = experts
            .iter()
            .map(|expert| expert.down.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let (expert_gate_up_ptrs, expert_down_ptrs) = if let Some(view) = tier_layer.as_ref() {
            (view.gate_up_ptrs_device(), view.down_ptrs_device())
        } else {
            let gate_ptrs = gate_entries
                .iter()
                .flat_map(|entry| entry.to_ne_bytes())
                .collect::<Vec<_>>();
            let down_ptrs = down_entries
                .iter()
                .flat_map(|entry| entry.to_ne_bytes())
                .collect::<Vec<_>>();
            let expert_gate_up_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)
            {
                Ok(tensor) => tensor,
                Err(error) => return Err(error.into()),
            };
            if let Err(error) = gpu.memcpy_htod_auto(&expert_gate_up_ptrs.buf, &gate_ptrs) {
                let _ = gpu.free_tensor(expert_gate_up_ptrs);
                return Err(error.into());
            }
            let expert_down_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
                Ok(tensor) => tensor,
                Err(error) => {
                    let _ = gpu.free_tensor(expert_gate_up_ptrs);
                    return Err(error.into());
                }
            };
            if let Err(error) = gpu.memcpy_htod_auto(&expert_down_ptrs.buf, &down_ptrs) {
                let _ = gpu.free_tensor(expert_gate_up_ptrs);
                let _ = gpu.free_tensor(expert_down_ptrs);
                return Err(error.into());
            }
            (expert_gate_up_ptrs, expert_down_ptrs)
        };
        let tier_identity = tier_layer
            .as_ref()
            .map(|view| {
                let lease = view.lease();
                let placement: Vec<u8> = (0..config.num_experts)
                    .map(
                        |eid| match view.placement(eid).expect("validated tier entry") {
                            crate::expert_tier_plan::Placement::Device => 0,
                            crate::expert_tier_plan::Placement::HostMapped => 1,
                        },
                    )
                    .collect();
                hipfire_dispatch::families::moe::ExpertTierIdentity::new(
                    lease.generation(),
                    // The view's own digest: the logical digest for Resident
                    // (byte-identical to before), the BorrowPrefill digest otherwise.
                    view.view_digest(),
                    &placement,
                    view.codec_table(),
                    view.gate_up_ptrs(),
                    view.down_ptrs(),
                    lease.keepalive(),
                )
                .map_err(|message| invalid(format!("tier identity: {message}")))
            })
            .transpose()?;
        let identity = NEXT_EXPERT_SET_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let expert_views = ExpertViewSet {
            experts: &experts,
            identity,
            host_mapped: local_host_mapped,
            tier_identity: tier_identity.as_ref(),
        };
        if let Err(error) = cache.bind_live(
            &table,
            &expert_views,
            &expert_gate_up_ptrs,
            &expert_down_ptrs,
            None,
            None,
        ) {
            if tier_layer.is_none() {
                let _ = gpu.free_tensor(expert_gate_up_ptrs);
                let _ = gpu.free_tensor(expert_down_ptrs);
            }
            return Err(Qwen4GpuForwardError::Dispatch(format!(
                "expert live binding: {error:?}"
            )));
        }
        let cpu_admission = cpu_splice::CpuSpliceAdmission {
            // `qwen4.expert_keep` payloads are compact (kept experts only), so the
            // original-id indexed CPU owners can never address them: pruned
            // layers stay single-device fully resident.
            enabled: hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled()
                && allow_tiers
                && layer_label < config.num_hidden_layers
                && kept_ids.is_none(),
            host_mapped: local_host_mapped && down_source.buf.is_host_mapped(),
            ep: weights.ep_placement().is_some(),
            slab: false,
            tier: tier_layer.is_some(),
            awq: experts.iter().any(|expert| {
                expert.gate_up.dispatch_ref().awq_scale.is_some()
                    || expert.down.dispatch_ref().awq_scale.is_some()
            }),
            supported: experts.iter().all(|expert| {
                hipfire_dispatch::cpu_exec::cpu_quant_for(expert.gate_up.dtype).is_some()
                    && hipfire_dispatch::cpu_exec::cpu_quant_for(expert.down.dtype).is_some()
            }),
        };
        let cpu_expert_owners = cpu_admission
            .armed()
            .then(|| {
                cpu_splice::CpuExpertOwners::new(
                    gpu,
                    gate_up_source,
                    down_source,
                    config.num_experts,
                    gate_up_expert_bytes,
                    down_expert_bytes,
                )
            })
            .flatten();
        let cpu_deferred_experts = cpu_expert_owners.is_some();
        // D2 split: its own admission, distinct from the whole-layer D1
        // splice above, whose layers it never shares (one layer never carries
        // both numerical modes). A hot-set tier layer is requested by
        // `memory.offload_exec=cpu` (passback keeps D1 behaviour) or by this
        // load's per-load default ([`crate::d2_default`]); an un-tiered
        // host-mapped aggregate layer only by that default (the explicit `cpu`
        // mode keeps D1 there, unchanged). The default never reaches D1 or the
        // process-wide mode. It has no arch gate: a tier arms on any plan the
        // loader admitted (`expert_hotset::hotset_arch_admits`: default sidecar
        // on gfx1201 only, an explicit sidecar path on any arch), and every
        // route it has no masked ids or cold-slot installation for is refused
        // below with one line.
        let explicit_cpu_split = hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled()
            && !hipfire_dispatch::cpu_exec::passback_enabled();
        let whole_layer = tier_layer.is_none()
            && !cpu_deferred_experts
            && local_host_mapped
            && down_source.buf.is_host_mapped();
        let split_kind = match tier_layer.as_ref() {
            Some(_) => Some(cpu_splice::SplitOwnerKind::Tier),
            None => whole_layer.then_some(cpu_splice::SplitOwnerKind::WholeLayer),
        };
        let split_admission = cpu_splice::CpuSplitAdmission {
            enabled: d2_split_enabled(
                tier_layer.is_some(),
                explicit_cpu_split,
                d2_default,
                allow_tiers,
                layer_label < config.num_hidden_layers,
                kept_ids.is_none(),
                head_split,
            ),
            kind: split_kind,
            bound_hotset: tier_layer
                .as_ref()
                .is_some_and(|view| view.lease().hotset_sidecar_sha256().is_some()),
            host_experts: match tier_layer.as_ref() {
                Some(view) => (0..config.num_experts)
                    .filter(|&eid| {
                        view.placement(eid) == Some(crate::expert_tier_plan::Placement::HostMapped)
                    })
                    .count(),
                None if whole_layer => config.num_experts,
                None => 0,
            },
            flash_next: config.profile.is_flash_next(),
            cpu_executed: experts.iter().all(|expert| {
                expert.gate_up.dtype == DType::MQ4G256V2 && expert.down.dtype == DType::MQ4G128V2
            }),
            ep: cpu_admission.ep,
            awq: cpu_admission.awq,
            fp8: fp8_tables.is_some(),
            narrow_stage: expert_stage_takes_narrow_windows(
                &gpu.arch,
                crate::expert_residency::expert_stage_placement_enabled(&gpu.arch),
                hipfire_config::developer_var(QWEN4_EXPERT_STAGE_MIN_ROWS_ENV)
                    .ok()
                    .as_deref(),
            ),
        };
        if borrow_view {
            // A BorrowPrefill runtime serves only wide prompt tiles over host
            // mirrors of the donors; it never carries a CPU splice or D2 split
            // (admission refuses them; the measured D2 default is decided only
            // per load, so it is checked here).
            if split_admission.armed() {
                return Err(invalid(format!(
                    "layer {layer_label}: decode-share BorrowPrefill runtime cannot arm a CPU \
                     expert split (set memory.offload_exec=pcie / HIPFIRE_QWEN4_D2_DEFAULT=0 to \
                     load with decode-share)"
                )));
            }
        } else if let Some(refusal) = split_admission.refusal() {
            cpu_splice::log_split_refusal_once(refusal, &gpu.arch);
        }
        // The owner is a set of borrowed views over memory the loader already
        // owns (the tier's pinned host owner, or the aggregate allocations'
        // pinned bytes): no expert is copied or allocated a second time.
        let cpu_cold_owners = if split_admission.armed() {
            let owners = match tier_layer.as_ref() {
                Some(view) => cpu_splice::CpuTierOwners::new(
                    &view.lease(),
                    view.layer(),
                    config.num_experts,
                    gate_up_expert_bytes,
                    down_expert_bytes,
                )
                .map(cpu_splice::HostSpanOwner::Tier),
                None => cpu_splice::CpuExpertOwners::new(
                    gpu,
                    gate_up_source,
                    down_source,
                    config.num_experts,
                    gate_up_expert_bytes,
                    down_expert_bytes,
                )
                .map(cpu_splice::HostSpanOwner::WholeLayer)
                .ok_or_else(|| {
                    "the aggregate expert allocations have no resolvable pinned host bytes"
                        .to_owned()
                }),
            };
            match owners {
                Ok(owners) => Some(owners),
                Err(why) => {
                    eprintln!(
                        "  [moe-cpu-split] qwen4 layer {layer_label}: {why}; this layer \
                         falls back to the PCIe mapped-expert read"
                    );
                    None
                }
            }
        } else {
            None
        };
        let runtime = Self {
            table,
            cache,
            router,
            shared_scalar,
            shared_gate,
            shared_up,
            shared_down,
            experts_all_gate_up_mq4: gate_up_dtype == DType::MQ4G256V2,
            experts,
            identity,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            expert_gate_up_entries: gate_entries,
            expert_down_entries: down_entries,
            symmetric: false,
            fp8: fp8_tables.is_some(),
            host_mapped: local_host_mapped,
            cpu_deferred_experts,
            cpu_expert_owners,
            cpu_cold_owners,
            cpu_split: None,
            tier: tier_layer.as_ref().map(|view| view.lease()),
            tier_identity,
            slab: None,
            label: layer_label,
            retained_identities: Vec::new(),
            expert_keep_mask: upload_expert_keep_mask(
                gpu,
                config,
                if moe.gate.name.starts_with("mtp.") {
                    config.num_hidden_layers
                } else {
                    layer_label
                },
            )?,
        }
        .seal_retained_identities()?;
        // The executor owns device storage, so it is created last: nothing
        // after it can fail and strand it.
        runtime.attach_cpu_split(
            gpu,
            config,
            if head_split {
                1
            } else {
                hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS
            },
        )
    }

    /// Create the layer's D2 split executor from its cold-expert owner's
    /// ownership mask (`None` owner: nothing to attach). The executor owns
    /// device storage, so this is the last fallible step of a layer build and a
    /// failure releases the whole layer.
    fn attach_cpu_split(
        mut self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        rows: usize,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let Some(placement) = self
            .cpu_cold_owners
            .as_ref()
            .map(|owners| owners.placement())
        else {
            return Ok(self);
        };
        // `HIPFIRE_QWEN4_D2_RETURN`, `HIPFIRE_QWEN4_D2_READY` and
        // `HIPFIRE_QWEN4_D2_JOIN` are read from the immutable process snapshot
        // for this load: an invalid spelling refuses the load before any
        // executor storage exists. `copy`, `sync` and `host` leave the executor
        // exactly as `new` built it; they are the kill values, and unset
        // return/join are `copy`/`host` except on gfx1100, where the arch
        // default requests `mapped`/`cpwait` (`default_d2_transport`; an
        // explicit value always wins). A refused `mapped` or `cpwait` is
        // capability routing, logged here and summarized once per load by the
        // loader, never silent.
        // The initial ctor keeps the legacy default-4 executor rows; the
        // load-only post-admission prewarm (`reserve_d2_wide_rows`) resizes an
        // armed layer to the real need before the first window.
        let transport = hipfire_config::memory::qwen4_d2_transport(&gpu.arch).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("layer {} D2 attach: {error}", self.label))
        })?;
        let (request, join) = (transport.ret, transport.join);
        let ready = hipfire_config::memory::qwen4_d2_ready().map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("layer {} D2 attach: {error}", self.label))
        })?;
        let built = self.new_split_executor(gpu, config, &placement, rows, request, ready, join);
        match built {
            Ok(executor) => {
                self.cpu_split = Some(executor);
                Ok(self)
            }
            Err(error) => {
                let label = self.label;
                let cleanup = self
                    .free_gpu(gpu)
                    .map(|free| format!("; release also failed: {free:?}"))
                    .unwrap_or_default();
                Err(Qwen4GpuForwardError::Dispatch(format!(
                    "layer {label} CPU split executor: {error}{cleanup}"
                )))
            }
        }
    }

    /// Single-device runtime of a trunk layer of an expert-slab artifact: every
    /// expert is a view into the layer's one arena (resident or host-mapped) at
    /// the offsets of the validated span table, with its own dtype, and the
    /// pointer tables name those views. Residency and codec come from the
    /// layer's own arena and per-expert table, never from expert 0.
    ///
    /// `split_requested`: the D2 split was asked for this load (explicit
    /// `memory.offload_exec=cpu` or the per-load default). A host-mapped arena
    /// whose present experts all have a CPU decoder then serves decode and
    /// `<=4`-row verify of every routed slot on the CPU, straight from the
    /// arena's pinned host bytes (the slab span kind of the D2 owner).
    #[allow(clippy::too_many_arguments)]
    fn build_slab(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        arena_ref: &TensorRef,
        layer_label: usize,
        config: &Qwen4Config,
        layout: &Qwen4ExpertTableLayout<'_>,
        [router, shared_scalar, shared_gate, shared_up, shared_down]: [ProjectionView; 5],
        split_requested: bool,
    ) -> Result<Self, Qwen4GpuForwardError> {
        use crate::expert_arena::{ARENA_TAIL_PAD_BYTES, CODEC_MQ4};
        use crate::expert_slabs::ExpertFormat;
        if !matches!(layout, Qwen4ExpertTableLayout::Single) {
            return Err(invalid(format!(
                "layer {layer_label}: slab arenas are single-device only (no EP route)"
            )));
        }
        if !rdna_compute::qwen4_slab_codec_admitted(&gpu.arch, CODEC_MQ4) {
            return Err(invalid(format!(
                "layer {layer_label}: the expert-slab route is gfx1100/gfx1151/gfx1201-only, device is {}",
                gpu.arch
            )));
        }
        let plan = match &weights.manifest.identity.recipe {
            crate::weights::Qwen4TierRecipe::FlashNextSlab { arenas, .. } => arenas
                .layer(layer_label)
                .ok_or_else(|| invalid(format!("expert arena plan has no layer {layer_label}")))?,
            _ => {
                return Err(invalid(format!(
                    "layer {layer_label} declares an expert arena under a non-slab recipe"
                )))
            }
        };
        let (rows_gate_up, hidden, mi) = (
            2 * config.moe_intermediate_size,
            config.hidden_size,
            config.moe_intermediate_size,
        );
        if config.num_experts != plan.experts.len()
            || [rows_gate_up, hidden, mi] != [1280, 2560, 640]
        {
            return Err(invalid(format!(
                "layer {layer_label}: the expert slab geometry is 512 experts of [1280,2560] / [2560,640]"
            )));
        }
        let arena = weights.resident(arena_ref)?;
        let host_mapped = arena.buf.is_host_mapped();
        if arena.buf.size() < plan.bytes {
            return Err(invalid(format!(
                "layer {layer_label} arena holds {} bytes, plan needs {}",
                arena.buf.size(),
                plan.bytes
            )));
        }
        // A resident arena is exact-fit plus its zeroed pad; a host-mapped one
        // carries `upload_raw_host_mapped`'s own larger zeroed tail.
        if !host_mapped && arena.buf.size() < plan.bytes + ARENA_TAIL_PAD_BYTES {
            return Err(invalid(format!(
                "layer {layer_label} resident arena has no zeroed {ARENA_TAIL_PAD_BYTES}-byte tail pad"
            )));
        }
        let codecs = plan
            .codec_table()
            .map_err(|error| invalid(format!("layer {layer_label}: {error}")))?;
        // The layer's single cold codec (its W3 or E8P experts', or MQ4 = no cold
        // bucket for an MQ4-only layer): refuses a codec this arch does not serve
        // and a layer holding both W3 and E8P. The planner table below is
        // membership (cold vs MQ4) and relies on it; the kernels bind to it.
        let cold_codec = plan
            .cold_codec(&gpu.arch)
            .map_err(|error| invalid(format!("layer {layer_label}: {error}")))?;
        // Absent (pruned, never routed) experts carry MQ4 dtypes so they never name a cold codec.
        let formats: Vec<ExpertFormat> = plan
            .experts
            .iter()
            .map(|expert| expert.effective_format())
            .collect();
        let dtypes_of = |format: ExpertFormat| match format {
            ExpertFormat::Mq2I4 => Err(invalid("MQ2-I4 has no expert-slab route")),
            format => Ok(format.dtypes()),
        };
        let (gate_up_source_name, down_source_name) = match &weights
            .layer(layer_label)
            .and_then(|layer| layer.ffn.moe())
            .map(|moe| &moe.expert_source_layout)
        {
            Some(ExpertSourceLayout::PackedFused { gate_up, down, .. }) => {
                (gate_up.clone(), down.clone())
            }
            _ => {
                return Err(invalid(format!(
                    "layer {layer_label} uses an unsupported expert source layout"
                )))
            }
        };
        let mut experts = Vec::with_capacity(config.num_experts);
        let mut metadata = Vec::with_capacity(config.num_experts);
        let mut gate_up_dtypes = Vec::with_capacity(config.num_experts);
        let mut down_dtypes = Vec::with_capacity(config.num_experts);
        for (expert, (entry, format)) in plan.experts.iter().zip(&formats).enumerate() {
            let (gate_up_dtype, down_dtype) = dtypes_of(*format)?;
            let (gate_up_bytes, down_bytes) = (format.gate_up_bytes(), format.down_bytes());
            let absent = expert_pruned(config.kept_expert_ids(layer_label, false), expert)
                && entry.gate_up_offset == entry.down_offset;
            if !absent
                && (entry.gate_up_offset + gate_up_bytes != entry.down_offset
                    || entry.down_offset + down_bytes > plan.bytes)
            {
                return Err(invalid(format!(
                    "layer {layer_label} expert {expert} extents leave the arena"
                )));
            }
            let gate_up = if absent {
                ProjectionView::absent(gate_up_dtype, rows_gate_up, hidden)
            } else {
                ProjectionView::from_packed(
                    arena,
                    entry.gate_up_offset,
                    gate_up_bytes,
                    gate_up_dtype,
                    rows_gate_up,
                    hidden,
                )
            };
            let down = if absent {
                ProjectionView::absent(down_dtype, hidden, mi)
            } else {
                ProjectionView::from_packed(
                    arena,
                    entry.down_offset,
                    down_bytes,
                    down_dtype,
                    hidden,
                    mi,
                )
            };
            // Planar W3/E8P dtypes have no row stride: resource headers take
            // their code-row bytes (3*k/8 and k/4); MQ4 uses its group-row stride.
            let stride = |dtype: DType, k: usize| match dtype {
                DType::MQ3I4G128R256 | DType::MQ3I4G128R128 => k * 3 / 8,
                DType::E8PI4G128R256 | DType::E8PI4G128R128 => k / 4,
                other => row_stride(other, k),
            };
            let resource = |name: &str, shape: Vec<usize>, dtype: DType, bytes: usize, k: usize| {
                ExpertResource::new(
                    name,
                    hipfire_dispatch::pipeline::sealed_moe::SLAB_RESOURCE_FORMAT,
                    shape,
                    dtype,
                    bytes,
                    stride(dtype, k),
                    16,
                    dtype_rotation_plan(dtype),
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
                })
            };
            let resources = ExpertResources::fused(
                // One source name per distinct representation: the table refuses a
                // name repeated with different metadata.
                resource(
                    &format!("{gate_up_source_name}#{gate_up_dtype:?}"),
                    vec![rows_gate_up, hidden],
                    gate_up_dtype,
                    gate_up_bytes,
                    hidden,
                )?,
                resource(
                    &format!("{down_source_name}#{down_dtype:?}"),
                    vec![hidden, mi],
                    down_dtype,
                    down_bytes,
                    mi,
                )?,
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resources: {error:?}"))
            })?;
            let record = ExpertMetadata::new(expert, 0, expert, resources).map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert metadata: {error:?}"))
            })?;
            metadata.push(if absent { record.into_absent() } else { record });
            gate_up_dtypes.push(gate_up_dtype);
            down_dtypes.push(down_dtype);
            experts.push(Qwen4ExpertView { gate_up, down });
        }
        // The D2 slab span entries: the same plan resolution the D2 default's
        // real-arm probe reads ([`cpu_slab_owner::plan_entries`]).
        let span_entries =
            cpu_slab_owner::plan_entries(plan, config.kept_expert_ids(layer_label, false))
                .map_err(|error| invalid(format!("layer {layer_label}: {error}")))?;
        let table = ExpertTable::new(metadata)
            .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("expert table: {error:?}")))?;
        let mut cache = table
            .prepare_binding(0, 1, gpu.device_id)
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert binding: {error:?}"))
            })?;
        let gate_entries = experts
            .iter()
            .map(|expert| expert.gate_up.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let down_entries = experts
            .iter()
            .map(|expert| expert.down.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let mixed = codecs.iter().any(|codec| *codec != codecs[0]);
        // The tag tensor of a mixed layer ([`slab_codec_tag_bytes`]): u8 planner
        // membership, then the real codecs as i32 (the table-driven tile mask's
        // owner table); its logical shape is the u8 table the sealed validator sizes.
        let codec_tags = if mixed {
            let bytes = slab_codec_tag_bytes(&codecs);
            let mut tags = gpu.upload_raw(&bytes, &[bytes.len()])?;
            tags.shape = vec![codecs.len()];
            Some(tags)
        } else {
            None
        };
        let free_tags = |gpu: &mut Gpu, tags: Option<GpuTensor>| {
            if let Some(tags) = tags {
                let _ = gpu.free_tensor(tags);
            }
        };
        let to_bytes = |entries: &[usize]| {
            entries
                .iter()
                .flat_map(|entry| entry.to_ne_bytes())
                .collect::<Vec<u8>>()
        };
        let expert_gate_up_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                free_tags(gpu, codec_tags);
                return Err(error.into());
            }
        };
        let expert_down_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                free_tags(gpu, codec_tags);
                let _ = gpu.free_tensor(expert_gate_up_ptrs);
                return Err(error.into());
            }
        };
        let upload = gpu
            .memcpy_htod_auto(&expert_gate_up_ptrs.buf, &to_bytes(&gate_entries))
            .and_then(|()| gpu.memcpy_htod_auto(&expert_down_ptrs.buf, &to_bytes(&down_entries)));
        if let Err(error) = upload {
            free_tags(gpu, codec_tags);
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(error.into());
        }
        let identity = NEXT_EXPERT_SET_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let expert_views = ExpertViewSet {
            experts: &experts,
            identity,
            host_mapped,
            tier_identity: None,
        };
        if let Err(error) = cache.bind_live(
            &table,
            &expert_views,
            &expert_gate_up_ptrs,
            &expert_down_ptrs,
            None,
            codec_tags.as_ref(),
        ) {
            free_tags(gpu, codec_tags);
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(Qwen4GpuForwardError::Dispatch(format!(
                "expert live binding: {error:?}"
            )));
        }
        // The K1 planner and entries load before any capture.
        gpu.ensure_qwen4_slab_kernels(cold_codec)?;
        gpu.reserve_qwen4_slab(0)?;
        // D2 slab span: the same admission every owner kind takes. A slab
        // layer has no EP, AWQ or FP8 route (refused above), and a resident
        // arena has no host-placed expert.
        let present = span_entries.iter().filter(|entry| !entry.absent);
        let split_admission = cpu_splice::CpuSplitAdmission {
            enabled: split_requested && layer_label < config.num_hidden_layers,
            kind: host_mapped.then_some(cpu_splice::SplitOwnerKind::Slab),
            bound_hotset: false,
            host_experts: if host_mapped {
                present.clone().count()
            } else {
                0
            },
            flash_next: config.profile.is_flash_next(),
            cpu_executed: present.clone().all(|entry| {
                hipfire_dispatch::cpu_exec::cpu_quant_for(entry.gate_up_dtype).is_some()
                    && hipfire_dispatch::cpu_exec::cpu_quant_for(entry.down_dtype).is_some()
            }),
            ep: false,
            awq: false,
            fp8: false,
            narrow_stage: expert_stage_takes_narrow_windows(
                &gpu.arch,
                crate::expert_residency::expert_stage_placement_enabled(&gpu.arch),
                hipfire_config::developer_var(QWEN4_EXPERT_STAGE_MIN_ROWS_ENV)
                    .ok()
                    .as_deref(),
            ),
        };
        if let Some(refusal) = split_admission.refusal() {
            cpu_splice::log_split_refusal_once(refusal, &gpu.arch);
        }
        // Borrowed views of the owning arena's pinned host bytes: no expert is
        // copied or allocated a second time.
        let cpu_cold_owners = if split_admission.armed() {
            match cpu_slab_owner::CpuSlabOwners::new(
                gpu,
                arena,
                plan.bytes,
                span_entries.iter().copied(),
            ) {
                Ok(owners) => Some(cpu_splice::HostSpanOwner::Slab(owners)),
                Err(why) => {
                    eprintln!(
                        "  [moe-cpu-split] qwen4 layer {layer_label}: {why}; this layer \
                         falls back to the PCIe mapped-expert read"
                    );
                    None
                }
            }
        } else {
            None
        };
        let runtime = Self {
            table,
            cache,
            router,
            shared_scalar,
            shared_gate,
            shared_up,
            shared_down,
            experts_all_gate_up_mq4: gate_up_dtypes
                .iter()
                .all(|dtype| *dtype == DType::MQ4G256V2),
            experts,
            identity,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            expert_gate_up_entries: gate_entries,
            expert_down_entries: down_entries,
            symmetric: false,
            fp8: false,
            host_mapped,
            cpu_deferred_experts: false,
            cpu_expert_owners: None,
            cpu_cold_owners,
            cpu_split: None,
            tier: None,
            tier_identity: None,
            label: layer_label,
            retained_identities: Vec::new(),
            expert_keep_mask: upload_expert_keep_mask(gpu, config, layer_label)?,
            slab: Some(Qwen4SlabLayer {
                cold_codec,
                codecs,
                gate_up_dtypes,
                down_dtypes,
                codec_tags,
                arena_base: arena.buf.as_ptr() as usize,
                arena_bytes: plan.bytes,
            }),
        }
        .seal_retained_identities()?;
        // The executor owns device storage, so it is created last.
        runtime.attach_cpu_split(gpu, config, hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS)
    }

    /// Load-only D2 prewarm/reserve: replace this layer's default-4 executor
    /// with one sized for the REAL requested rows (from
    /// `Qwen4Bundle::native_mtp_capture_rows`, passed by the loader
    /// post-admission), preserving the Return/Ready/Join selections. The
    /// caller (C: bundle/load) owns the memory charge and calls this BEFORE
    /// the first window — never with a window in flight (the old executor is
    /// freed first, so a temporary peak of old+new is charged, never hot
    /// growth). `requested_rows` is clamped to `4..=STORAGE`; unarmed layers
    /// (no cold owner / no executor) are a no-op returning `false`, as is an
    /// executor that already covers the request. Returns whether this layer
    /// was resized. Existing pure [`Self::min_split_capacity_rows`] stays the
    /// actual-instance capacity reader for Bundle/Rate (no Vec, no alloc).
    pub(crate) fn reserve_d2_wide_rows(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        requested_rows: usize,
    ) -> Result<bool, Qwen4GpuForwardError> {
        if self.cpu_cold_owners.is_none() || self.cpu_split.is_none() {
            return Ok(false);
        }
        use hipfire_dispatch::cpu_split::{CPU_SPLIT_MAX_ROWS, CPU_SPLIT_STORAGE_MAX_ROWS};
        let rows = requested_rows.clamp(CPU_SPLIT_MAX_ROWS, CPU_SPLIT_STORAGE_MAX_ROWS);
        let covered = self
            .cpu_split
            .as_ref()
            .is_some_and(|executor| executor.capacity_rows() >= rows);
        if covered {
            return Ok(false);
        }
        // Return/Ready/Join are load-immutable process-snapshot selections:
        // re-reading them here yields exactly what the initial ctor applied,
        // so the resized executor preserves them. Snapshot refusal before any
        // new storage exists (same as the initial path).
        let transport = hipfire_config::memory::qwen4_d2_transport(&gpu.arch).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("layer {} D2 prewarm: {error}", self.label))
        })?;
        let (request, join) = (transport.ret, transport.join);
        let ready = hipfire_config::memory::qwen4_d2_ready().map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("layer {} D2 prewarm: {error}", self.label))
        })?;
        let placement = self
            .cpu_cold_owners
            .as_ref()
            .map(|owners| owners.placement())
            .expect("checked above");
        let old = self.cpu_split.take().expect("checked above");
        if let Err(error) = old.free_gpu(gpu) {
            return Err(Qwen4GpuForwardError::Dispatch(format!(
                "layer {} D2 prewarm release: {error:?}",
                self.label
            )));
        }
        let label = self.label;
        let fresh = self
            .new_split_executor(gpu, config, &placement, rows, request, ready, join)
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("layer {label} D2 prewarm: {error}"))
            })?;
        // Re-apply is inside new_split_executor; publish under the same log lines.
        self.cpu_split = Some(fresh);
        Ok(true)
    }

    /// Shared executor construction behind [`Self::attach_cpu_split`] (initial
    /// default path) and [`Self::reserve_d2_wide_rows`] (post-admission resize):
    /// build for `rows` with explicit Return/Ready/Join requests. Refused
    /// `mapped`/`cpwait` keep capability routing (logged, never silent).
    fn new_split_executor(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        placement: &[u8],
        rows: usize,
        request: hipfire_config::memory::D2Return,
        ready: hipfire_config::memory::D2Ready,
        join: hipfire_config::memory::D2Join,
    ) -> Result<hipfire_dispatch::cpu_split::CpuSplitExecutor, String> {
        use hipfire_config::memory::{D2Join, D2Ready, D2Return};
        let mut executor = hipfire_dispatch::cpu_split::CpuSplitExecutor::new(
            gpu,
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts_per_tok,
            rows,
            placement,
        )
        .map_err(|error| format!("{error:?}"))?;
        let release = |executor: hipfire_dispatch::cpu_split::CpuSplitExecutor, error: String| {
            let release = executor
                .free_gpu(gpu)
                .err()
                .map(|free| format!("; executor release also failed: {free:?}"))
                .unwrap_or_default();
            format!("{error}{release}")
        };
        if request == D2Return::Mapped {
            if let Err(error) = executor.enable_mapped_return(gpu) {
                return Err(release(executor, format!("{error:?}")));
            }
            if let Some(reason) = executor.return_refusal() {
                eprintln!(
                    "  qwen4 D2 return: layer {}: mapped refused, copy kept: {reason}",
                    self.label
                );
            }
        }
        if ready == D2Ready::Event {
            if let Err(error) = executor.enable_event_ready() {
                return Err(release(executor, format!("{error:?}")));
            }
        }
        if join == D2Join::CpWait {
            if let Err(error) = executor.enable_cpwait_join(gpu) {
                return Err(release(executor, format!("{error:?}")));
            }
            if let Some(reason) = executor.join_refusal() {
                eprintln!(
                    "  qwen4 D2 join: layer {}: cpwait refused, host kept: {reason}",
                    self.label
                );
            }
        }
        Ok(executor)
    }

    /// Check every routed expert header of this layer for the symmetric grid
    /// the IU4 route needs and record the verdict. QT44 gate/up and QT53
    /// down only; one asymmetric or non-finite header refuses the layer.
    fn verify_symmetric(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
    ) -> Result<bool, Qwen4GpuForwardError> {
        if self.fp8 {
            // qt=55 blocks have no QT44/QT53 headers: nothing to check, and
            // the layer can never declare the symmetric variant.
            self.symmetric = false;
            return Ok(false);
        }
        if self.expert_keep_mask.is_some() && self.qt44_qt53_formats() {
            let gu: Vec<u8> = self
                .expert_gate_up_entries
                .iter()
                .filter(|&&p| p != 0)
                .flat_map(|p| p.to_ne_bytes())
                .collect();
            let dn: Vec<u8> = self
                .expert_down_entries
                .iter()
                .filter(|&&p| p != 0)
                .flat_map(|p| p.to_ne_bytes())
                .collect();
            let gu_table = gpu.upload_raw(&gu, &[gu.len()])?;
            let dn_table = match gpu.upload_raw(&dn, &[dn.len()]) {
                Ok(table) => table,
                Err(error) => {
                    let _ = gpu.free_tensor(gu_table);
                    return Err(error.into());
                }
            };
            let result = (|| -> Result<bool, Qwen4GpuForwardError> {
                Ok(gpu.qwen4_moe_sym_check(
                    &gu_table,
                    2 * config.moe_intermediate_size,
                    config.hidden_size,
                    gu.len() / 8,
                    136,
                )? && gpu.qwen4_moe_sym_check(
                    &dn_table,
                    config.hidden_size,
                    config.moe_intermediate_size,
                    dn.len() / 8,
                    68,
                )?)
            })();
            let _ = gpu.free_tensor(gu_table);
            let _ = gpu.free_tensor(dn_table);
            self.symmetric = result?;
            return Ok(self.symmetric);
        }
        let formats = self.qt44_qt53_formats();
        self.symmetric = formats
            && gpu.qwen4_moe_sym_check(
                &self.expert_gate_up_ptrs,
                2 * config.moe_intermediate_size,
                config.hidden_size,
                config.num_experts,
                136,
            )?
            && gpu.qwen4_moe_sym_check(
                &self.expert_down_ptrs,
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                68,
            )?;
        Ok(self.symmetric)
    }

    /// Derive and keep the layer's retained-route identities
    /// ([`hipfire_dispatch::pipeline::sealed_moe::retained_route_identities`]).
    fn seal_retained_identities(mut self) -> Result<Self, Qwen4GpuForwardError> {
        self.retained_identities = self
            .cache
            .retained_route_identities(&self.table, self.label as u32)
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("retained route identity: {error:?}"))
            })?;
        Ok(self)
    }

    /// The plain facts [`classify_ep_lane_expert`] reads of this runtime.
    fn ep_lane_facts(&self) -> EpLaneRuntimeFacts {
        EpLaneRuntimeFacts {
            label: self.label,
            host_mapped: self.host_mapped,
            packed_mq4: !self.experts.is_empty() && self.qt44_qt53_formats(),
            plain: !self.cpu_deferred_experts
                && self.cpu_expert_owners.is_none()
                && self.cpu_cold_owners.is_none()
                && self.cpu_split.is_none()
                && self.tier.is_none()
                && self.tier_identity.is_none()
                && self.slab.is_none()
                && !self.fp8
                && self.expert_keep_mask.is_none(),
        }
    }

    /// Every routed expert is the QT44 gate/up + QT53 down pair (`MQ4G256V2`/`MQ4G128V2`) the
    /// grouped GEMMs take.
    fn qt44_qt53_formats(&self) -> bool {
        self.experts.iter().all(|expert| {
            expert.gate_up.dtype == DType::MQ4G256V2 && expert.down.dtype == DType::MQ4G128V2
        })
    }

    /// The layer may take the owner-routed F16 WMMA grouped route (exact gfx1100): QT44/QT53
    /// experts, no stock-FP8 blocks. The WMMA arms read no symmetric header, so this replaces
    /// [`Self::verify_symmetric`] as the per-layer verdict there.
    fn f16_wmma_tile_eligible(&self) -> bool {
        !self.fp8 && self.qt44_qt53_formats()
    }
    /// Device bytes [`Self::from_moe`] allocates: the gate/up and down
    /// pointer tables, one device pointer per expert each.
    pub(crate) fn device_bytes(config: &Qwen4Config) -> Option<usize> {
        let tables = Self::allocation_sizes(config)?;
        usize::try_from(
            tables
                .iter()
                .try_fold(0u64, |total, &b| total.checked_add(b))?,
        )
        .ok()
    }

    /// [`Self::device_bytes`] per allocation: the gate/up and the down
    /// pointer table, `num_experts` 8-byte pointers each (omitted when the
    /// profile has no experts).
    pub(crate) fn allocation_sizes(config: &Qwen4Config) -> Option<Vec<u64>> {
        let table =
            u64::try_from(config.num_experts.checked_mul(std::mem::size_of::<u64>())?).ok()?;
        // The charge is two tables; doubling must not overflow either.
        table.checked_mul(2)?;
        Some(if table == 0 {
            Vec::new()
        } else {
            vec![table, table]
        })
    }

    pub(crate) fn free_gpu(mut self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        // D2 split executor first: it drains its queued uploads/masks before
        // freeing. If that drain cannot be proven, nothing of this layer is
        // released: the executor retains its own owners and the runtime
        // (tier lease, expert owners, pointer tables) is leaked with it, so
        // no model reader is dropped under in-flight GPU work.
        if let Some(split) = self.cpu_split.take() {
            if let Err(error) = split.free_gpu(gpu) {
                std::mem::forget(self);
                return Some(error);
            }
        }
        // Tier tables are frozen borrowed sub-views; only the lease frees them.
        if self.tier.is_some() {
            return self
                .expert_keep_mask
                .and_then(|mask| gpu.free_tensor(mask).err());
        }
        let mut first = None;
        let tags = self.slab.and_then(|slab| slab.codec_tags);
        for tensor in [self.expert_gate_up_ptrs, self.expert_down_ptrs]
            .into_iter()
            .chain(tags)
            .chain(self.expert_keep_mask)
        {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

fn upload_expert_keep_mask(
    gpu: &mut Gpu,
    config: &Qwen4Config,
    layer: usize,
) -> Result<Option<GpuTensor>, Qwen4GpuForwardError> {
    let Some(keep) = &config.expert_keep else {
        return Ok(None);
    };
    let ids = if layer < config.num_hidden_layers {
        keep.kept(layer)
    } else {
        let Some(ids) = keep.mtp_kept() else {
            return Ok(None);
        };
        ids
    };
    let mut mask = vec![0u8; config.num_experts];
    for &id in ids {
        mask[id as usize] = 1;
    }
    Ok(Some(gpu.upload_raw(&mask, &[mask.len()])?))
}

/// What a donor layer's VRAM holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DonorState {
    /// Its own experts, ordered for every later compute-stream launch.
    Resident,
    /// A host layer's experts are (or may be partly) copied over it.
    Overwritten,
    /// Its restore is issued on the copy stream and recorded on `ready`, but
    /// the compute stream has not yet waited on it.
    Restoring,
}

/// Long-prefill host-expert DMA pipeline (G2) on borrowed VRAM. The last
/// resident expert layer of each parity is that parity's stage: once its own
/// MoE has run, the host layers of the same parity are copied over it in
/// turn (L+2 refill), and after the last of them its bytes are restored
/// from a pinned host copy. Each fork and join is a recorded event, and
/// the forward makes the compute stream wait on the restores before it
/// returns. Any later launch on that stream (the active stream is installed
/// once, by `ensure_capture_stream`, and never replaced) therefore sees the
/// donor layers whole. Routing-independent: neither the copies nor the
/// pointer tables depend on the experts a row picks.
pub(crate) struct Qwen4ExpertStage {
    /// Forwards of at least this many rows stage.
    min_rows: usize,
    /// Where `min_rows` came from (`HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` or the default).
    min_rows_source: StageMinRowsSource,
    /// Long forwards with a wide-hidden capture stage too
    /// ([`QWEN4_EXPERT_STAGE_CAPTURE_ENV`]).
    capture: bool,
    /// Donor trunk layer per parity.
    donors: [usize; 2],
    /// Borrowed device views of each donor's gate/up and down experts,
    /// parity-major. Never freed here: the bundle's weights own them.
    donor_views: Vec<GpuTensor>,
    /// Pinned host copies of the same bytes: the restore source.
    mirrors: Vec<GpuTensor>,
    /// Static per-expert pointer tables into each donor's gate/up and down
    /// VRAM, parity-major. They never replace a layer's bound mapped tables.
    tables: Vec<GpuTensor>,
    /// EP tables preserve non-owned pointers from the runtime being bound.
    compact_tables: Vec<GpuTensor>,
    compact_keys: Vec<usize>,
    full_keys: Vec<std::cell::OnceCell<usize>>,
    compact_bound: Vec<std::cell::Cell<bool>>,
    first_host: usize,
    placement: Option<Qwen4EpPlacement>,
    bytes: [usize; 2],
    copy_stream: hip_bridge::Stream,
    ready: Vec<hip_bridge::Event>,
    free: Vec<hip_bridge::Event>,
    /// Sticky until the donors are back: a failed forward leaves it set,
    /// and every later forward restores first ([`Self::restore_blocking`]).
    state: std::cell::Cell<[DonorState; 2]>,
    /// `Some` for a slab-arena stage: per-layer rebased tables (in `tables`)
    /// and the host sources ([`slab_stage`]). Its stage buffers are either
    /// dedicated VRAM buffers (in `mirrors`) or, when `borrowed`, two
    /// resident slab layers (`donors`, `donor_views`) restored from pinned
    /// copies in `mirrors`.
    slab: Option<slab_stage::SlabStageMeta>,
}

/// S1 slow-link layer-major prefill transport: the bounded expert page ring
/// and the layer-ahead weight schedule ([`crate::slowlink_prefill`]).
pub(crate) mod panel_stage;
pub(crate) mod slab_stage;

impl Qwen4ExpertStage {
    const EP_ENV: &'static str = "HIPFIRE_EP_EXPERT_STAGE";

    /// Per-forward admission: full EP (EP2..EP8) chunk tables read other ranks' donor VRAM
    /// and must never stage. S1 supplies true only when every trunk MoE layer
    /// binds the compact owner-routed route. Single-card admission is unchanged.
    ///
    /// Entering a compact staged chunk after a full chunk is safe: the donor
    /// overwrite waits on local `free`, recorded after this chunk's local
    /// waits on peer ready; peer ready follows the previous forward in the
    /// peer compute stream, so its earlier donor reads have completed.
    fn begin_forward(&self, ep_compact_all_layers: bool) -> bool {
        self.placement.is_none() || ep_compact_all_layers
    }

    /// Before any full-table EP chunk, S1 calls this on EVERY rank, then
    /// crosses its host barrier before ANY rank launches that chunk. `join`
    /// marks donors Resident after enqueueing restore waits, not host completion,
    /// so even Resident donors must be drained before peers can read them.
    fn ep_quiesce(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if gpu
            .active_stream
            .as_ref()
            .map(|stream| gpu.hip.stream_is_capturing(stream))
            .transpose()?
            .unwrap_or(false)
        {
            return Err(invalid("G2 EP quiescence cannot run during capture"));
        }
        gpu.hip.device_synchronize()?;
        self.restore_blocking(gpu)
    }

    fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        chunk_rows: usize,
        host_sources: Option<&ep_reserve_forward::EpStageHostSources>,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        if let Some(placement) = bundle.weights.ep_placement() {
            if !rdna_compute::expert_capabilities::stage_policy(&gpu.arch).expert_parallel_stage {
                eprintln!(
                    "qwen4 EP rank {}/{}: expert staging not admitted (needs gfx1201)",
                    placement.rank, placement.ranks
                );
                return Ok(None);
            }
        }
        if !crate::expert_residency::expert_stage_enabled(gpu) {
            return Ok(None);
        }
        // Settle the host link (probe or cache hit) before any stage buffer
        // takes VRAM: the `min_rows` default reads it, and the 64 MiB probe
        // must not compete with the buffers. gfx1201 never reads it.
        if rdna_compute::expert_capabilities::stage_reads_link(&gpu.arch) {
            let _ = crate::host_link::host_link(gpu);
        }
        let Some(first_host) = moe
            .iter()
            .position(|layer| layer.as_ref().is_some_and(RoutedExpertWeights::host_mapped))
        else {
            return Ok(None);
        };
        // Slab arenas and tier layers: dedicated VRAM stage buffers take fresh
        // VRAM that only the probe-free placement switch charges to the frozen
        // placement (`expert_stage_placement_enabled`: explicit `1`, or
        // gfx1201). A stage the measured link alone turned on (a narrow
        // discrete link, e.g. gfx1100) may not allocate bytes no placement
        // reserved, and must not give resident layers up for
        // them, so a slab arena load borrows two resident slab layers as the
        // stage instead (restored from pinned host copies, like the legacy
        // donors); a tier layout has no borrowed form and keeps the zero-copy
        // mapped reads.
        if moe.iter().flatten().any(|layer| layer.slab.is_some())
            || bundle.weights.expert_tiers().is_some()
        {
            let kind = slab_stage::slab_stage_kind(
                crate::expert_residency::expert_stage_placement_enabled(&gpu.arch),
                bundle.weights.expert_tiers().is_some(),
            );
            return match kind {
                slab_stage::SlabStageKind::Dedicated => {
                    Self::new_slab(gpu, bundle, moe, first_host, chunk_rows, false)
                }
                slab_stage::SlabStageKind::Borrowed => {
                    Self::new_slab(gpu, bundle, moe, first_host, chunk_rows, true)
                }
                slab_stage::SlabStageKind::Off => {
                    eprintln!(
                        "  qwen4 long-prefill expert staging: off (dedicated tier stage buffers are not charged to placement on {} unless {QWEN4_EXPERT_STAGE_ENV}=1; zero-copy mapped reads)",
                        gpu.arch
                    );
                    hipfire_config::load_summary::note(
                        hipfire_config::load_summary::Facet::Prefill,
                        format!(
                            "long-prefill expert staging off (tier stage buffers not charged to placement on {})",
                            gpu.arch
                        ),
                    );
                    Ok(None)
                }
            };
        }
        let config = &bundle.config;
        // G2's stage layout is the frozen Flash-Next QT44/QT53 geometry.
        // Other quantized artifacts keep their existing mapped path.
        if config.num_experts != 512
            || config.hidden_size != 2560
            || config.moe_intermediate_size != 640
            || moe[first_host.saturating_sub(2)..]
                .iter()
                .flatten()
                .any(|layer| {
                    layer.experts.iter().any(|expert| {
                        rdna_compute::expert_capabilities::ExpertQt::of_dtypes(
                            expert.gate_up.tensor.dtype,
                            expert.down.tensor.dtype,
                        ) != Some(rdna_compute::expert_capabilities::QT_MQ4_XTS)
                    })
                })
        {
            return Ok(None);
        }
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::Prefill,
                format!("long-prefill expert staging off ({reason})"),
            );
            Ok(None)
        };
        // Resident layers are a prefix (`place_routed_experts`), so each
        // donor's own MoE runs before every host layer of its parity.
        if first_host < 2
            || !moe[first_host..]
                .iter()
                .all(|layer| layer.as_ref().is_some_and(RoutedExpertWeights::host_mapped))
        {
            return off("needs two resident expert layers ahead of every host layer");
        }
        let mut donors = [first_host - 2, first_host - 1];
        donors.sort_by_key(|layer| layer % 2);
        let placement = bundle.weights.ep_placement();
        let ranks = placement.map_or(1, |placement| placement.ranks);
        let bytes = stage_donor_bytes(
            config.num_experts,
            placement.map_or(0, |placement| placement.rank),
            ranks,
        );
        // The mirrors of all ranks share one host RAM/GTT budget: charge every rank at the
        // largest (rank 0) share so an uneven split is never under-counted.
        let layer_bytes = stage_donor_bytes(config.num_experts, 0, ranks)
            .iter()
            .sum::<usize>();
        let mirror_bytes = crate::expert_residency::expert_stage_mirror_bytes(layer_bytes, ranks)
            .map_err(invalid)?;
        let admitted = crate::expert_residency::check_host_ram(
            mirror_bytes,
            rdna_compute::kv_slots::mem_available_bytes(),
            crate::expert_residency::ttm_pool_estimate(),
        )
        .and_then(|()| {
            crate::expert_residency::check_gtt_cap(
                mirror_bytes,
                crate::expert_residency::gtt_budget(),
            )
        });
        if let Err(reason) = admitted {
            return off(&format!("restore copies: {reason}"));
        }
        let parts =
            |layer: usize| -> Result<[(&TensorRef, usize, usize); 2], Qwen4GpuForwardError> {
                let weights = bundle
                    .weights
                    .layer(layer)
                    .ok_or_else(|| invalid("G2 stage layer is not owned"))?
                    .ffn
                    .moe()
                    .ok_or_else(|| invalid("G2 stage layer has no MoE FFN"))?;
                Ok([
                    (
                        &weights.experts_gate_up,
                        bytes[0],
                        QWEN4_EXPERT_STAGE_GATE_UP_BYTES,
                    ),
                    (
                        &weights.experts_down,
                        bytes[1],
                        QWEN4_EXPERT_STAGE_DOWN_BYTES,
                    ),
                ])
            };
        for layer in first_host..moe.len() {
            if let Some([gate, down]) = host_sources.and_then(|sources| sources.pair(layer)) {
                // EP reserve: this scratch-donor layer's host source is the loan's mirror.
                if gate.len() < bytes[0] || down.len() < bytes[1] {
                    return off("a loan mirror has no pinned host bytes of the stage size");
                }
                continue;
            }
            for (source, bytes, _) in parts(layer)? {
                let owner = bundle.weights.resident(source)?;
                // Host addresses come only from owning tensors. Borrowed
                // expert views are never re-tagged as HostMapped (G1).
                if gpu.host_bytes(owner).is_none_or(|host| host.len() < bytes) {
                    return off("a host expert owner has no pinned host bytes of the stage size");
                }
            }
        }
        for layer in donors {
            for (source, bytes, _) in parts(layer)? {
                let owner = bundle.weights.resident(source)?;
                if owner.buf.is_host_mapped() || owner.buf.size() < bytes {
                    return off("a donor expert owner is not device memory of the stage size");
                }
            }
        }
        gpu.ensure_capture_stream()?;
        let (min_rows, min_rows_source) = qwen4_expert_stage_min_rows(gpu, chunk_rows);
        let mut stage = Self {
            min_rows,
            min_rows_source,
            capture: hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_CAPTURE_ENV, true),
            donors,
            donor_views: Vec::with_capacity(4),
            mirrors: Vec::with_capacity(4),
            tables: Vec::with_capacity(4),
            compact_tables: Vec::new(),
            compact_keys: if placement.is_some() {
                moe.iter()
                    .map(|runtime| {
                        runtime.as_ref().map_or(0, |runtime| {
                            runtime.expert_gate_up_ptrs.buf.as_ptr() as usize
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            },
            full_keys: if placement.is_some() {
                (0..moe.len()).map(|_| std::cell::OnceCell::new()).collect()
            } else {
                Vec::new()
            },
            compact_bound: if placement.is_some() {
                (0..moe.len())
                    .map(|_| std::cell::Cell::new(false))
                    .collect()
            } else {
                Vec::new()
            },
            first_host,
            placement,
            bytes,
            copy_stream: gpu.hip.stream_create_non_blocking()?,
            ready: Vec::with_capacity(2),
            free: Vec::with_capacity(2),
            state: std::cell::Cell::new([DonorState::Resident; 2]),
            slab: None,
        };
        let result = (|| {
            for _ in 0..2 {
                stage.ready.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
                stage.free.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
            }
            for layer in donors {
                for (source, bytes, stride) in parts(layer)? {
                    let owner = bundle.weights.resident(source)?;
                    let view = owner.sub_offset(0, bytes);
                    // The mirror reads the loaded bytes back once, at load.
                    let mut host = vec![0u8; bytes];
                    gpu.memcpy_dtoh_auto(&mut host, &view.buf)?;
                    stage
                        .mirrors
                        .push(gpu.upload_raw_host_mapped(&host, &[bytes])?);
                    drop(host);
                    let base = view.buf.as_ptr() as usize;
                    if placement.is_none() {
                        let entries = (0..config.num_experts)
                            .flat_map(|expert| (base + expert * stride).to_ne_bytes())
                            .collect::<Vec<_>>();
                        stage
                            .tables
                            .push(gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?);
                        gpu.memcpy_htod_auto(
                            &stage.tables.last().expect("table just allocated").buf,
                            &entries,
                        )?;
                    }
                    stage.donor_views.push(view);
                }
            }
            if placement.is_some() {
                for (layer, runtime) in moe
                    .iter()
                    .enumerate()
                    .skip(first_host)
                    .filter_map(|(layer, runtime)| runtime.as_ref().map(|runtime| (layer, runtime)))
                {
                    for (kind, entries) in [
                        &runtime.expert_gate_up_entries,
                        &runtime.expert_down_entries,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let redirected = stage.redirect_entries(layer, kind, entries);
                        stage
                            .compact_tables
                            .push(gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?);
                        gpu.memcpy_htod_auto(
                            &stage.compact_tables.last().expect("allocated").buf,
                            &redirected,
                        )?;
                        stage
                            .tables
                            .push(gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?);
                    }
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            let _ = stage.free_gpu(gpu);
            return Err(error);
        }
        eprintln!(
            "  qwen4 long-prefill expert staging: forwards of >= {} rows stage host expert layers {first_host}..{} through resident layers {} and {} (borrowed donor VRAM, no dedicated stage buffers; {} MiB pinned restore copies, host RAM on top of the placement's pinned experts; wide-hidden capture forwards {}; min_rows from {}, prefill chunk {chunk_rows})",
            stage.min_rows,
            moe.len(),
            donors[0],
            donors[1],
            2 * layer_bytes >> 20,
            if stage.capture { "stage" } else { "mapped" },
            stage.min_rows_source.label()
        );
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::Prefill,
            format!(
                "long-prefill expert staging on (>= {} rows)",
                stage.min_rows
            ),
        );
        Ok(Some(stage))
    }

    /// The stage tables a host `layer` reads from on a staged forward.
    fn tables(&self, layer: usize) -> MoeStageTables<'_> {
        if self.slab.is_some() {
            return self.slab_tables(layer);
        }
        let parity = layer % 2;
        if self.placement.is_some() && self.compact_bound[layer].get() {
            return self.tables_compact(layer);
        }
        let index = if self.placement.is_some() {
            layer - self.first_host
        } else {
            parity
        };
        MoeStageTables {
            gate_up: &self.tables[2 * index],
            down: &self.tables[2 * index + 1],
            free: &self.free[parity],
        }
    }

    /// S1 binds this accessor when its compact owner-routed runtime is used.
    fn tables_compact(&self, layer: usize) -> MoeStageTables<'_> {
        if self.placement.is_none() {
            return self.tables(layer);
        }
        let index = 2 * (layer - self.first_host);
        MoeStageTables {
            gate_up: &self.compact_tables[index],
            down: &self.compact_tables[index + 1],
            free: &self.free[layer % 2],
        }
    }

    fn redirect_entries(&self, layer: usize, kind: usize, entries: &[usize]) -> Vec<u8> {
        let placement = self.placement.expect("EP tables");
        let base = self.donor_views[2 * (layer % 2) + kind].buf.as_ptr() as usize;
        let stride = [
            QWEN4_EXPERT_STAGE_GATE_UP_BYTES,
            QWEN4_EXPERT_STAGE_DOWN_BYTES,
        ][kind];
        redirect_owned_entries(placement, base, stride, entries)
    }

    /// Initialize a full EP table once, before any capture/tape. Compact
    /// tables are built at load and remain separate from peer-mapped tables.
    fn prepare_runtime(
        &self,
        gpu: &Gpu,
        layer: usize,
        runtime: Option<&Qwen4MoeLayerRuntime>,
    ) -> Result<(), Qwen4GpuForwardError> {
        if self.placement.is_none() || layer < self.first_host {
            return Ok(());
        }
        let runtime = runtime.ok_or_else(|| invalid("G2 staged EP runtime missing"))?;
        let key = runtime.expert_gate_up_ptrs.buf.as_ptr() as usize;
        let compact = key == self.compact_keys[layer];
        self.compact_bound[layer].set(compact);
        if compact {
            return Ok(());
        }
        if let Some(&prepared) = self.full_keys[layer].get() {
            return if prepared == key {
                Ok(())
            } else {
                Err(invalid("G2 full runtime changed"))
            };
        }
        if gpu
            .active_stream
            .as_ref()
            .map(|stream| gpu.hip.stream_is_capturing(stream))
            .transpose()?
            .unwrap_or(false)
        {
            return Err(invalid(
                "G2 EP stage table initialization cannot run during capture",
            ));
        }
        for (kind, entries) in [
            &runtime.expert_gate_up_entries,
            &runtime.expert_down_entries,
        ]
        .into_iter()
        .enumerate()
        {
            let redirected = self.redirect_entries(layer, kind, entries);
            gpu.memcpy_htod_auto(
                &self.tables[2 * (layer - self.first_host) + kind].buf,
                &redirected,
            )?;
        }
        self.full_keys[layer]
            .set(key)
            .map_err(|_| invalid("G2 full runtime initialized twice"))?;
        Ok(())
    }

    fn set_state(&self, parity: usize, value: DonorState) {
        let mut state = self.state.get();
        state[parity] = value;
        self.state.set(state);
    }

    /// Copy host `layer`'s experts over its parity's donor once the stage is
    /// free, recording `ready`.
    fn enqueue_layer(
        &self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        layer: usize,
        host_sources: Option<&ep_reserve_forward::EpStageHostSources>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let parity = layer % 2;
        if let Some([gate, down]) = host_sources.and_then(|sources| sources.pair(layer)) {
            // EP reserve: a scratch-donor layer stages from the loan's pinned mirror.
            gpu.hip
                .stream_wait_event(&self.copy_stream, &self.free[parity])?;
            self.set_state(parity, DonorState::Overwritten);
            return self.copy_into_donor(gpu, parity, [gate, down]);
        }
        let weights = bundle
            .weights
            .layer(layer)
            .ok_or_else(|| invalid("G2 stage layer is not owned"))?
            .ffn
            .moe()
            .ok_or_else(|| invalid("G2 stage layer has no MoE FFN"))?;
        let gate = gpu
            .host_bytes(bundle.weights.resident(&weights.experts_gate_up)?)
            .ok_or_else(|| invalid("G2 gate/up pinned owner missing"))?;
        let down = gpu
            .host_bytes(bundle.weights.resident(&weights.experts_down)?)
            .ok_or_else(|| invalid("G2 down pinned owner missing"))?;
        gpu.hip
            .stream_wait_event(&self.copy_stream, &self.free[parity])?;
        // From the first byte issued, the donor no longer holds its experts.
        self.set_state(parity, DonorState::Overwritten);
        self.copy_into_donor(gpu, parity, [gate, down])
    }

    /// Copy the donor's own bytes back once the last host layer of its
    /// parity is done with the stage, recording `ready`.
    fn enqueue_restore(&self, gpu: &Gpu, parity: usize) -> Result<(), Qwen4GpuForwardError> {
        let [gate, down] = self.mirror_bytes(gpu, parity)?;
        gpu.hip
            .stream_wait_event(&self.copy_stream, &self.free[parity])?;
        self.copy_into_donor(gpu, parity, [gate, down])?;
        self.set_state(parity, DonorState::Restoring);
        Ok(())
    }

    fn copy_into_donor(
        &self,
        gpu: &Gpu,
        parity: usize,
        sources: [&[u8]; 2],
    ) -> Result<(), Qwen4GpuForwardError> {
        for (kind, (source, bytes)) in sources.into_iter().zip(self.bytes).enumerate() {
            gpu.hip.memcpy_htod_async(
                &self.donor_views[2 * parity + kind].buf,
                &source[..bytes],
                &self.copy_stream,
            )?;
        }
        gpu.hip
            .event_record(&self.ready[parity], Some(&self.copy_stream))?;
        Ok(())
    }

    fn mirror_bytes<'a>(
        &'a self,
        gpu: &Gpu,
        parity: usize,
    ) -> Result<[&'a [u8]; 2], Qwen4GpuForwardError> {
        let host = |kind: usize| {
            gpu.host_bytes(&self.mirrors[2 * parity + kind])
                .ok_or_else(|| invalid("G2 donor restore copy missing"))
        };
        Ok([host(0)?, host(1)?])
    }

    /// After `layer`'s MoE is issued on a staged forward: a donor frees its
    /// stage, and a donor or host layer refills its parity for the next host
    /// layer L+2.., or, past the last one, restores the donor.
    fn after_moe(
        &self,
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        layer: usize,
        host_sources: Option<&ep_reserve_forward::EpStageHostSources>,
    ) -> Result<(), Qwen4GpuForwardError> {
        if self.slab.is_some() {
            return self.slab_after_moe(gpu, moe, layer);
        }
        let parity = layer % 2;
        let donor = self.donors[parity] == layer;
        if !donor
            && !moe[layer]
                .as_ref()
                .is_some_and(RoutedExpertWeights::host_mapped)
        {
            return Ok(());
        }
        if donor {
            // A host layer's routed down records `free` itself.
            let compute = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| invalid("G2 compute stream missing"))?;
            gpu.hip.event_record(&self.free[parity], Some(compute))?;
        }
        match (layer + 2..moe.len()).step_by(2).find(|&next| {
            moe[next]
                .as_ref()
                .is_some_and(RoutedExpertWeights::host_mapped)
        }) {
            Some(next) => self.enqueue_layer(gpu, bundle, next, host_sources),
            None if self.state.get()[parity] == DonorState::Overwritten => {
                self.enqueue_restore(gpu, parity)
            }
            None => Ok(()),
        }
    }

    /// End of a staged forward: the compute stream waits on every restore,
    /// charging what the trunk did not hide to this forward.
    fn join(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if self.slab.is_some() {
            return self.slab_join(gpu);
        }
        for parity in 0..2 {
            if self.state.get()[parity] == DonorState::Overwritten {
                self.enqueue_restore(gpu, parity)?;
            }
            if self.state.get()[parity] == DonorState::Restoring {
                let compute = gpu
                    .active_stream
                    .as_ref()
                    .ok_or_else(|| invalid("G2 compute stream missing"))?;
                gpu.hip.stream_wait_event(compute, &self.ready[parity])?;
                self.set_state(parity, DonorState::Resident);
            }
        }
        Ok(())
    }

    /// Recovery when a staged forward did not reach [`Self::join`]: drain the
    /// device, then copy every donor not known resident back synchronously.
    /// The state stays set on failure, so the next forward retries before
    /// anything reads the donors.
    fn restore_blocking(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if self.slab.is_some() {
            // Dedicated and tier stage buffers never overwrite an owner; a
            // borrowed slab stage restores its donor arenas.
            return self.slab_restore_blocking(gpu);
        }
        if self.state.get() == [DonorState::Resident; 2] {
            return Ok(());
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            if gpu.hip.stream_is_capturing(stream)? {
                return Err(invalid(
                    "G2 donor expert layers need a restore, which a stream capture cannot run",
                ));
            }
        }
        gpu.hip.device_synchronize()?;
        for parity in 0..2 {
            if self.state.get()[parity] == DonorState::Resident {
                continue;
            }
            for (kind, source) in self.mirror_bytes(gpu, parity)?.into_iter().enumerate() {
                gpu.hip
                    .memcpy_htod(&self.donor_views[2 * parity + kind].buf, source)?;
            }
            gpu.hip.device_synchronize()?;
            self.set_state(parity, DonorState::Resident);
        }
        Ok(())
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = gpu.hip.stream_synchronize(&self.copy_stream).err();
        if let Err(error) = gpu.hip.stream_destroy(self.copy_stream) {
            first.get_or_insert(error);
        }
        for event in self.ready.into_iter().chain(self.free) {
            if let Err(error) = gpu.hip.event_destroy(event) {
                first.get_or_insert(error);
            }
        }
        for tensor in self
            .tables
            .into_iter()
            .chain(self.compact_tables)
            .chain(self.mirrors)
        {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// The G2 stage's host work around each MoE call of a staged forward, run by
/// the step interpreter (so the whole layer program stays one interpreter
/// call): a host layer's compute waits for its staged experts, and after the
/// call the stage is freed / refilled ([`Qwen4ExpertStage::after_moe`]).
struct StagedMoeHooks<'s, 'c> {
    stage: &'s Qwen4ExpertStage,
    bundle: &'s Qwen4Bundle,
    moe: &'s [Option<Qwen4MoeLayerRuntime>],
    host_sources: Option<&'s ep_reserve_forward::EpStageHostSources>,
    /// The layer of the next MoE call.
    layer: usize,
    /// The stage error behind a hook failure, returned instead of the
    /// interpreter's stringified one.
    error: Option<Qwen4GpuForwardError>,
    /// The forward's DFlash2 layer capture, announced every step.
    capture: Option<&'c mut (dyn MoeStepHooks + 'c)>,
}

impl StagedMoeHooks<'_, '_> {
    fn fail(&mut self, result: Result<(), Qwen4GpuForwardError>) -> Result<(), DispatchError> {
        result.map_err(|error| {
            let message = error.to_string();
            self.error = Some(error);
            DispatchError::Hip(message)
        })
    }
}

impl MoeStepHooks for StagedMoeHooks<'_, '_> {
    fn before_step(&mut self, gpu: &mut Gpu, step: &Step<'_>) -> Result<(), DispatchError> {
        match self.capture.as_deref_mut() {
            Some(capture) => capture.before_step(gpu, step),
            None => Ok(()),
        }
    }

    fn before_moe(&mut self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let result = match self.moe.get(self.layer) {
            None => Err(invalid("G2 staged forward ran more MoE calls than layers")),
            Some(moe) if !moe.as_ref().is_some_and(RoutedExpertWeights::host_mapped) => Ok(()),
            Some(_) => match gpu.active_stream.as_ref() {
                None => Err(invalid("G2 compute stream missing")),
                Some(compute) => gpu
                    .hip
                    .stream_wait_event(compute, &self.stage.ready[self.layer % 2])
                    .map_err(Qwen4GpuForwardError::from),
            },
        };
        self.fail(result)
    }

    fn after_moe(&mut self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        // Down recorded free before combine. Refill this parity for L+2 or
        // restore its donor; no routing results are involved.
        let result =
            self.stage
                .after_moe(gpu, self.bundle, self.moe, self.layer, self.host_sources);
        self.layer += 1;
        self.fail(result)
    }
}

/// Scratch views required by one sealed Qwen4 MoE invocation.  The caller owns
/// these reusable buffers; the sealed expert table and live pointer resources
/// remain in [`Qwen4MoeLayerRuntime`].
pub(crate) struct Qwen4MoeScratch<'a> {
    pub(crate) router_logits: &'a GpuTensor,
    pub(crate) scalar_buf: &'a GpuTensor,
    pub(crate) x_rot_local: &'a GpuTensor,
    pub(crate) gate_up_buf: &'a GpuTensor,
    pub(crate) gate_buf: &'a GpuTensor,
    pub(crate) up_buf: &'a GpuTensor,
    pub(crate) ffn_hidden: &'a GpuTensor,
    pub(crate) ffn_out: &'a GpuTensor,
    pub(crate) gate_batch: &'a GpuTensor,
    pub(crate) up_batch: &'a GpuTensor,
    pub(crate) rot_batch: &'a GpuTensor,
    pub(crate) topk_indices: &'a GpuTensor,
    pub(crate) topk_weights: &'a GpuTensor,
    pub(crate) down_expanded: &'a GpuTensor,
}

/// Seal the one-row decode MoE of a QT44/QT53 `runtime` for a caller's step
/// list (the MTP head); the caller clears `output` with a preceding
/// `Step::Clear`. A stock-FP8 runtime has no indexed decode route (its decode
/// is the grouped prefill route, sealed by `program::seal_moe_for_layer`) and
/// is refused here.
pub(crate) fn seal_moe_decode<'a>(
    ctx: &'a DispatchCtx,
    config: &Qwen4Config,
    layer_index: usize,
    runtime: &'a Qwen4MoeLayerRuntime,
    input: &'a GpuTensor,
    output: &'a GpuTensor,
    scratch: Qwen4MoeScratch<'a>,
    row_gather: Option<&'a dyn MoeRowGather>,
) -> Result<SealedMoeCall<'a>, Qwen4GpuForwardError> {
    if runtime.fp8 {
        return Err(invalid(
            "qt=55 FP8 experts have no indexed decode route; seal them through \
             program::seal_moe_for_layer (grouped prefill at every row count)",
        ));
    }
    let dtypes = MoeDtypes {
        router: runtime.router.dtype,
        shared: Some(MoeSharedDtypes {
            selector: runtime.shared_scalar.dtype,
            gate: runtime.shared_gate.dtype,
            up: runtime.shared_up.dtype,
            down: runtime.shared_down.dtype,
        }),
        experts_all_gate_up_mq4: runtime
            .experts
            .iter()
            .all(|expert| expert.gate_up.dtype == DType::MQ4G256V2),
        routed_gate_up: runtime
            .experts
            .first()
            .map(|expert| expert.gate_up.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_down: runtime
            .experts
            .first()
            .map(|expert| expert.down.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    let shared = MoeSharedWeights {
        selector: runtime.shared_scalar.dispatch_ref(),
        gate: runtime.shared_gate.dispatch_ref(),
        up: runtime.shared_up.dispatch_ref(),
        down: runtime.shared_down.dispatch_ref(),
    };
    let params = MoeParams {
        dtypes,
        mq2_lloyd: None,
        recipe: MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        },
        // Single-token decode/MTP binding: never the symmetric IU4 prefill arm.
        route_policy: Some(qwen4_route_policy(
            config,
            &shared,
            MoeRouteCapability::Qt44Qt53Grouped,
            None,
            runtime
                .expert_keep_mask
                .as_ref()
                .map_or(0, |mask| mask.buf.as_ptr() as usize),
        )),
        normalization: MoeNormalization::Provided,
        batch_size: 1,
        hidden: config.hidden_size,
        mi: config.moe_intermediate_size,
        k: config.num_experts_per_tok,
        n_exp: config.num_experts,
        norm_topk_prob: config.norm_topk_prob,
        selection_bias: None,
        x_rot_prerotated: false,
        defer_routed_combine: false,
        cpu_deferred_experts: runtime.cpu_deferred_experts,
        ep_mode: MoeEpMode::None,
        layer_idx: layer_index as u16,
        x_norm: input,
        x_residual: output,
        routed_out: None,
        skip_shared: false,
        router: runtime.router.dispatch_ref(),
        shared: Some(MoeSharedDecode {
            weights: shared,
            intermediate: config.shared_expert_intermediate_size,
            scalar: scratch.scalar_buf,
            gate_out: scratch.gate_buf,
            up_out: scratch.up_buf,
        }),
        expert_gate_up_ptrs: &runtime.expert_gate_up_ptrs,
        expert_down_ptrs: &runtime.expert_down_ptrs,
        expert_ptrs_host: None,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        routed_gate_up_k: config.hidden_size,
        routed_down_m: config.hidden_size,
        routed_down_k: config.moe_intermediate_size,
        routed_experts: runtime,
        routed_gate_up_paro: None,
        routed_down_paro: None,
        router_logits: scratch.router_logits,
        x_rot_local: scratch.x_rot_local,
        gate_up_buf: scratch.gate_up_buf,
        ffn_hidden: scratch.ffn_hidden,
        ffn_out: scratch.ffn_out,
        gate_batch: scratch.gate_batch,
        up_batch: scratch.up_batch,
        rot_batch: scratch.rot_batch,
        topk_indices: scratch.topk_indices,
        topk_weights: scratch.topk_weights,
        down_expanded: scratch.down_expanded,
        ep_row_gather: row_gather,
    };
    let bound = BoundMoeExperts::from_cache(&runtime.table, &runtime.cache)
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("bound MoE experts: {error:?}")))?;
    let seal_started = qwen4_profile_start();
    let sealed_result = seal_decode(bound, ctx, params);
    qwen4_profile_record(Qwen4ProfilePhase::MoeSeal, seal_started);
    sealed_result
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}")))
}

/// Every device field of [`Qwen4GpuForwardScratch`], in allocation order.
/// The order is the plan's index: `plan` refuses to push a slot out of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScratchField {
    TokenIds,
    EmbeddingRot,
    Embeddings,
    Streams,
    HcNormalized,
    HcLow,
    HcUp,
    HcMixed,
    HcGates,
    Rotation,
    Projection,
    Projection2,
    GdnA,
    GdnB,
    GdnGate,
    GdnBeta,
    GdnRecurrentOutput,
    GdnBf16,
    GdnZ,
    GdnOutput,
    QsaIndex,
    QsaQgate,
    QsaK,
    QsaV,
    QsaOutput,
    QsaSelected,
    AttentionOutput,
    PleStaged,
    PleRows,
    PleKey,
    PleValue,
    PleQuery,
    PleGated,
    PleNormed,
    PleNormedWide,
    PleOutput,
    RouterLogits,
    MoeXRot,
    MoeGateUp,
    MoeGate,
    MoeUp,
    MoeHidden,
    FfnOutput,
    MoeSharedOutput,
    MoeGateBatch,
    MoeUpBatch,
    MoeRotBatch,
    MoeTopkIndices,
    MoeTopkWeights,
    MoeDownExpanded,
    MoeScalar,
    MoeExpertTokenCounts,
    MoeExpertOffsets,
    MoeSortedSlotIndex,
    MoeExpertTileIds,
    MoeInversePerm,
    MoeYGateUpGrouped,
    MoeYDownGrouped,
    Logits,
}

impl ScratchField {
    pub(crate) const COUNT: usize = Self::Logits as usize + 1;
    const ALL: [Self; Self::COUNT] = [
        Self::TokenIds,
        Self::EmbeddingRot,
        Self::Embeddings,
        Self::Streams,
        Self::HcNormalized,
        Self::HcLow,
        Self::HcUp,
        Self::HcMixed,
        Self::HcGates,
        Self::Rotation,
        Self::Projection,
        Self::Projection2,
        Self::GdnA,
        Self::GdnB,
        Self::GdnGate,
        Self::GdnBeta,
        Self::GdnRecurrentOutput,
        Self::GdnBf16,
        Self::GdnZ,
        Self::GdnOutput,
        Self::QsaIndex,
        Self::QsaQgate,
        Self::QsaK,
        Self::QsaV,
        Self::QsaOutput,
        Self::QsaSelected,
        Self::AttentionOutput,
        Self::PleStaged,
        Self::PleRows,
        Self::PleKey,
        Self::PleValue,
        Self::PleQuery,
        Self::PleGated,
        Self::PleNormed,
        Self::PleNormedWide,
        Self::PleOutput,
        Self::RouterLogits,
        Self::MoeXRot,
        Self::MoeGateUp,
        Self::MoeGate,
        Self::MoeUp,
        Self::MoeHidden,
        Self::FfnOutput,
        Self::MoeSharedOutput,
        Self::MoeGateBatch,
        Self::MoeUpBatch,
        Self::MoeRotBatch,
        Self::MoeTopkIndices,
        Self::MoeTopkWeights,
        Self::MoeDownExpanded,
        Self::MoeScalar,
        Self::MoeExpertTokenCounts,
        Self::MoeExpertOffsets,
        Self::MoeSortedSlotIndex,
        Self::MoeExpertTileIds,
        Self::MoeInversePerm,
        Self::MoeYGateUpGrouped,
        Self::MoeYDownGrouped,
        Self::Logits,
    ];

    /// Pair execution splits before the mixer: `HcMixed` (the mixer input)
    /// and `HcGates` (the attention write's gates, from prepare's paired HC
    /// read or the previous fragment's MoE row fold; the BF16-stream write
    /// reads no `HcNormalized`) survive that split. Keep this exhaustive when
    /// adding fields.
    fn npu_bank_lifetime(self) -> NpuBankLifetime {
        use NpuBankLifetime as L;
        match self {
            Self::TokenIds | Self::EmbeddingRot | Self::Embeddings | Self::PleStaged => {
                L::Initialize
            }
            Self::Streams => L::Residual,
            Self::HcMixed | Self::HcGates => L::PreparedInput,
            Self::PleRows => L::StagedRows,
            Self::HcNormalized
            | Self::HcLow
            | Self::HcUp
            | Self::Rotation
            | Self::GdnBf16
            | Self::PleKey
            | Self::PleValue
            | Self::PleQuery
            | Self::PleGated
            | Self::PleNormed
            | Self::PleNormedWide
            | Self::PleOutput => L::PrepareAndFragment,
            Self::Projection
            | Self::Projection2
            | Self::GdnA
            | Self::GdnB
            | Self::GdnGate
            | Self::GdnBeta
            | Self::GdnRecurrentOutput
            | Self::GdnZ
            | Self::GdnOutput
            | Self::QsaIndex
            | Self::QsaQgate
            | Self::QsaK
            | Self::QsaV
            | Self::QsaOutput
            | Self::QsaSelected
            | Self::AttentionOutput
            | Self::RouterLogits
            | Self::MoeXRot
            | Self::MoeGateUp
            | Self::MoeGate
            | Self::MoeUp
            | Self::MoeHidden
            | Self::FfnOutput
            | Self::MoeSharedOutput
            | Self::MoeGateBatch
            | Self::MoeUpBatch
            | Self::MoeRotBatch
            | Self::MoeTopkIndices
            | Self::MoeTopkWeights
            | Self::MoeDownExpanded
            | Self::MoeScalar
            | Self::MoeExpertTokenCounts
            | Self::MoeExpertOffsets
            | Self::MoeSortedSlotIndex
            | Self::MoeExpertTileIds
            | Self::MoeInversePerm
            | Self::MoeYGateUpGrouped
            | Self::MoeYDownGrouped => L::Fragment,
            Self::Logits => L::Finish,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NpuBankLifetime {
    Initialize,
    Residual,
    PreparedInput,
    StagedRows,
    PrepareAndFragment,
    Fragment,
    Finish,
}

impl NpuBankLifetime {
    fn private(self) -> bool {
        matches!(
            self,
            Self::Residual | Self::PreparedInput | Self::StagedRows
        )
    }
}

/// Which pair executor owns bank B. Under `Npu` the in-projection outputs live
/// in the per-bank NPU C arenas across the switch; under `Emulated` the GPU
/// twin projection writes them into scratch one layer ahead of their Core, so
/// bank B keeps its own copies instead of sharing bank A's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NpuBankKind {
    Npu,
    Emulated,
}

impl NpuBankKind {
    /// Whether bank B owns `field` (rather than borrowing bank A's).
    fn owns(self, field: ScratchField) -> bool {
        use ScratchField as F;
        field.npu_bank_lifetime().private()
            || (self == Self::Emulated
                && matches!(
                    field,
                    F::Projection
                        | F::GdnA
                        | F::GdnB
                        | F::GdnZ
                        | F::QsaIndex
                        | F::QsaQgate
                        | F::QsaK
                        | F::QsaV
                ))
    }
}

/// How a scratch bank holds its device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScratchBackingKind {
    /// Every owning root is a `gpu.zeros` allocation (the ordinary bank).
    Owned,
    /// Owning roots are allocated, but `TokenIds` aliases a persistent tensor
    /// that another owner frees once (the resident bank of a decode-share load).
    ExternalIds,
    /// Every root, and `TokenIds`, is a view of external memory (a decode-share
    /// prefill bank over the donor span): the bank frees nothing.
    Borrowed,
}

impl ScratchBackingKind {
    /// Whether the bank frees the tensor of `field` (`slot` is its plan entry).
    fn frees(
        self,
        field: ScratchField,
        slot: &ScratchSlot,
        bank_kind: Option<NpuBankKind>,
    ) -> bool {
        match self {
            Self::Borrowed => false,
            Self::ExternalIds if field == ScratchField::TokenIds => false,
            Self::Owned | Self::ExternalIds => {
                slot.owner.is_none()
                    && slot.elements != 0
                    && bank_kind.map_or(true, |kind| kind.owns(field))
            }
        }
    }
}

/// The fields a bank with `slots` frees on teardown (a pure function of its
/// plan and backing, so the decode-share free rules are testable on the CPU).
#[cfg(test)]
fn scratch_free_list(
    slots: &[ScratchSlot],
    bank_kind: Option<NpuBankKind>,
    backing: ScratchBackingKind,
) -> Vec<ScratchField> {
    slots
        .iter()
        .enumerate()
        .filter(|(index, slot)| backing.frees(ScratchField::ALL[*index], slot, bank_kind))
        .map(|(index, _)| ScratchField::ALL[index])
        .collect()
}

/// One field of a scratch plan.
///
/// `owner == None`: the field owns a device allocation of `elements` of
/// `dtype`.  `owner == Some(o)`: the field is a borrowed, never-freed view of
/// `o`'s allocation from offset zero with the same `dtype` and an honest
/// `elements`; `o` itself always owns its allocation.
///
/// `pitch` is set for every member of an alias set (the owner included, except
/// a pure-capacity owner such as the grouped-down buffer): the row pitch in
/// elements that a lane segment base uses.  All members of one alias set share
/// the owner's pitch, so segments of different members that are live at once
/// occupy disjoint, identically addressed row ranges (see
/// [`Qwen4GpuForwardScratch::segment_view`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchSlot {
    pub elements: usize,
    pub dtype: DType,
    pub owner: Option<ScratchField>,
    pub pitch: Option<usize>,
    /// Elements into the owner's allocation where this view starts; zero for
    /// owners and every `Full`-mode slot.
    pub offset: usize,
}

impl ScratchSlot {
    fn bytes(&self) -> Result<usize, Qwen4GpuForwardError> {
        self.elements
            .checked_mul(self.dtype.size())
            .ok_or_else(|| invalid("scratch field byte size overflows"))
    }
}

fn scratch_mul(a: usize, b: usize) -> Result<usize, Qwen4GpuForwardError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("scratch size arithmetic overflows"))
}

/// Rows of PLE convolution input the compact scratch keeps after a forward:
/// the widest forward whose normed rows are read after it returns, for a
/// verify rollback.  A verify window is at most
/// `QWEN4_VERIFY_MAX_ROWS` rows (the row capture refuses more, `state.rs`
/// `row_capture_allocation_sizes_scoped`; the n-gram lane's block is
/// `NGRAM_DEPTHS + 1` of them; a mesh row repair's tokens equal its verify
/// epoch's rows).  A lane step (`forward_lanes`) is `lanes * rows` with at
/// most `lane::MAX_LANES` lanes of one store and a verify lane at most
/// `LANE_MAX_ROWS` rows (`forward_lanes` refuses more; an AR lane is one), and
/// a lane reads its window at its first row of the step.
pub(crate) const QWEN4_PLE_NORMED_KEEP_ROWS: usize = {
    let lanes = crate::lane::MAX_LANES * crate::gpu_forward_lanes::LANE_MAX_ROWS;
    if lanes > hipfire_config::QWEN4_VERIFY_MAX_ROWS {
        lanes
    } else {
        hipfire_config::QWEN4_VERIFY_MAX_ROWS
    }
};

/// Elements (256 bytes of F32) every phase-arena field start is rounded up to.
const COMPACT_ARENA_ALIGN: usize = 64;

fn round_up_arena(elements: usize) -> Result<usize, Qwen4GpuForwardError> {
    elements
        .checked_add(COMPACT_ARENA_ALIGN - 1)
        .map(|sum| sum / COMPACT_ARENA_ALIGN * COMPACT_ARENA_ALIGN)
        .ok_or_else(|| invalid("compact scratch arena overflows"))
}

/// The step-local field groups of the `FnSymGfx1201` phase arena, each as
/// `(field, elements, pitch)` in layout order: GDN, QSA, HC, PLE, MoE,
/// embedding.  Groups never run at once, so each lays out from element 0 of
/// the shared allocation.  `slots` supplies the MoE element counts, which the
/// plan sizes for the symmetric IU4 route.  The first GDN entry,
/// `Projection`, is the arena owner.
pub(crate) fn compact_arena_groups(
    config: &Qwen4Config,
    max_chunk: usize,
    slots: &[ScratchSlot],
) -> Result<Vec<Vec<(ScratchField, usize, Option<usize>)>>, Qwen4GpuForwardError> {
    use ScratchField as F;
    let layout = Qwen4ScratchLayout::for_rows(program_dims(config), max_chunk)
        .map_err(|error| invalid(error.to_string()))?;
    let n = max_chunk;
    let heads = config.linear_num_value_heads;
    let kv_width = scratch_mul(config.num_key_value_heads, config.head_dim)?;
    let mut groups = Vec::with_capacity(6);
    let sized = |fields: &[(F, usize)]| -> Result<_, Qwen4GpuForwardError> {
        fields
            .iter()
            .map(|&(field, width)| Ok((field, scratch_mul(n, width)?, Some(width))))
            .collect::<Result<Vec<_>, Qwen4GpuForwardError>>()
    };
    groups.push(sized(&[
        (F::Projection, layout.gdn_qkv),
        (F::Projection2, layout.gdn_qkv),
        (F::GdnRecurrentOutput, layout.gdn_value),
        (F::GdnZ, layout.gdn_value),
        (F::GdnOutput, layout.gdn_value),
        (F::GdnA, heads),
        (F::GdnB, heads),
        (F::GdnGate, heads),
        (F::GdnBeta, heads),
    ])?);
    let mut qsa = sized(&[
        (F::QsaQgate, 2 * layout.q_width),
        (F::QsaOutput, layout.q_width),
        (F::QsaIndex, layout.index_width),
        (F::QsaK, kv_width),
        (F::QsaV, kv_width),
    ])?;
    // The per-row selections live only inside the indexed-attention launch
    // (selection kernel then the attention kernels over it; never across a
    // step boundary): the last QSA member, a Raw byte slot viewed as the F32
    // elements that cover it.
    qsa.push((
        F::QsaSelected,
        slots[F::QsaSelected as usize]
            .elements
            .div_ceil(std::mem::size_of::<f32>()),
        None,
    ));
    groups.push(qsa);
    groups.push(sized(&[
        (F::HcNormalized, layout.wide),
        (F::HcLow, config.hc_lowrank),
        (F::HcUp, layout.wide),
    ])?);
    groups.push(sized(&[
        (F::PleQuery, layout.wide),
        (F::PleKey, layout.wide),
        (F::PleOutput, layout.wide),
        (F::PleValue, layout.hidden),
        (F::PleNormedWide, layout.wide),
    ])?);
    groups.push(
        [
            F::MoeYDownGrouped,
            F::MoeYGateUpGrouped,
            F::MoeDownExpanded,
            F::MoeXRot,
            F::MoeSharedOutput,
            F::MoeGateUp,
            F::MoeGate,
            F::MoeUp,
            F::MoeHidden,
            F::MoeScalar,
            F::MoeGateBatch,
            F::MoeUpBatch,
            F::MoeRotBatch,
        ]
        .iter()
        .map(|&field| (field, slots[field as usize].elements, None))
        .collect(),
    );
    groups.push(
        [F::Embeddings, F::EmbeddingRot]
            .iter()
            .map(|&field| (field, slots[field as usize].elements, None))
            .collect(),
    );
    Ok(groups)
}

/// `(offset, len)` in elements of lane segment `[begin, begin + rows)` of an
/// alias-set member whose owner row pitch is `pitch` (`logical_width <=
/// pitch` elements per row are live) inside an owner view of `owner_numel`
/// elements.  The base is `begin * pitch`, never `begin * logical_width`, so
/// two members of one set that are live for the same segment address the
/// same rows and segments of different sizes cannot overlap.  The returned
/// length is the segment's whole reservation, `rows * pitch`.
pub(crate) fn owner_pitch_extent(
    pitch: usize,
    owner_numel: usize,
    begin: usize,
    rows: usize,
    logical_width: usize,
) -> Result<(usize, usize), Qwen4GpuForwardError> {
    if logical_width > pitch {
        return Err(invalid(format!(
            "lane segment row width {logical_width} exceeds its owner row pitch {pitch}"
        )));
    }
    let offset = scratch_mul(begin, pitch)?;
    let len = scratch_mul(rows, pitch)?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid("lane segment reservation overflows"))?;
    if end > owner_numel {
        return Err(invalid(format!(
            "lane segment rows {begin}..{} at owner pitch {pitch} exceed the {owner_numel}-element owner",
            begin.saturating_add(rows)
        )));
    }
    Ok((offset, len))
}

/// Device buffers shared by one forward object.  All allocations happen in
/// `new`/`new_npu_bank`; token/chunk execution only creates borrowed subviews.
///
/// In [`Qwen4ForwardScratchMode::FnSymGfx1201`] the step-local fields of the
/// GDN, QSA, HC, PLE, MoE and embedding steps are views of the `Projection`
/// arena (see [`Self::plan`]) and `PleGated` views `Rotation`; only owners
/// are allocated, charged by [`Self::device_bytes`] and freed.
pub struct Qwen4GpuForwardScratch {
    pub max_chunk: usize,
    pub mode: Qwen4ForwardScratchMode,
    /// The plan this scratch was built from, in [`ScratchField`] order.
    pub(crate) slots: Vec<ScratchSlot>,
    /// NPU bank B borrows every layer-local field from bank A; only the
    /// bank-private lifetimes (and, under `emulate`, the in-projection outputs
    /// its twin projection writes ahead of the consuming Core) are owned here.
    /// Bank A outlives this view.
    bank_kind: Option<NpuBankKind>,
    backing: ScratchBackingKind,
    pub token_ids: GpuTensor,
    pub embedding_rot: GpuTensor,
    pub embeddings: GpuTensor,
    pub streams: GpuTensor,
    pub hc_normalized: GpuTensor,
    pub hc_low: GpuTensor,
    pub hc_up: GpuTensor,
    pub hc_mixed: GpuTensor,
    pub hc_gates: GpuTensor,
    pub rotation: GpuTensor,
    pub projection: GpuTensor,
    pub projection2: GpuTensor,
    pub gdn_a: GpuTensor,
    pub gdn_b: GpuTensor,
    pub gdn_gate: GpuTensor,
    pub gdn_beta: GpuTensor,
    pub gdn_recurrent_output: GpuTensor,
    pub gdn_bf16: GpuTensor,
    pub gdn_z: GpuTensor,
    pub gdn_output: GpuTensor,
    pub qsa_index: GpuTensor,
    pub qsa_qgate: GpuTensor,
    pub qsa_k: GpuTensor,
    pub qsa_v: GpuTensor,
    pub qsa_output: GpuTensor,
    pub qsa_selected: GpuTensor,
    pub attention_output: GpuTensor,
    pub ple_staged: GpuTensor,
    pub ple_rows: GpuTensor,
    pub ple_key: GpuTensor,
    pub ple_value: GpuTensor,
    pub ple_query: GpuTensor,
    pub ple_gated: GpuTensor,
    /// The PLE convolution input rows a partial verify rollback reads after
    /// the forward: the whole `max_chunk` in `Full`, the first
    /// [`QWEN4_PLE_NORMED_KEEP_ROWS`] rows in the compact plan.
    pub ple_normed: GpuTensor,
    /// Compact plan, `max_chunk > QWEN4_PLE_NORMED_KEEP_ROWS`: the normed rows
    /// of a forward wider than the kept rows (a view of the phase arena).
    /// Never read after the forward.
    pub ple_normed_wide: GpuTensor,
    /// Whether the last forward wrote its normed rows to
    /// [`Self::ple_normed_wide`], so [`Self::ple_normed_for_rollback`] refuses.
    ple_normed_in_wide: std::sync::atomic::AtomicBool,
    pub ple_output: GpuTensor,
    pub router_logits: GpuTensor,
    pub moe_x_rot: GpuTensor,
    pub moe_gate_up: GpuTensor,
    pub moe_gate: GpuTensor,
    pub moe_up: GpuTensor,
    pub moe_hidden: GpuTensor,
    /// Separate shared-expert down output.  The public MoE output is the
    /// routed accumulator; aliasing these buffers makes the shared GEMV
    /// overwrite the routed contribution before the source-ordered add.
    pub moe_shared_output: GpuTensor,
    /// The layer's FFN output (MoE routed accumulator or dense down output)
    /// that the HC write reads.
    pub ffn_output: GpuTensor,
    pub moe_gate_batch: GpuTensor,
    pub moe_up_batch: GpuTensor,
    pub moe_rot_batch: GpuTensor,
    pub moe_topk_indices: GpuTensor,
    pub moe_topk_weights: GpuTensor,
    pub moe_down_expanded: GpuTensor,
    pub moe_scalar: GpuTensor,
    pub moe_expert_token_counts: GpuTensor,
    pub moe_expert_offsets: GpuTensor,
    pub moe_sorted_slot_index: GpuTensor,
    pub moe_expert_tile_ids: GpuTensor,
    pub moe_inverse_perm: GpuTensor,
    pub moe_y_gate_up_grouped: GpuTensor,
    pub moe_y_down_grouped: GpuTensor,
    pub logits: GpuTensor,
}

impl Qwen4GpuForwardScratch {
    /// [`Self::plan_scoped`] for the whole model: the historical plan.
    fn plan(
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<(Vec<ScratchSlot>, usize), Qwen4GpuForwardError> {
        Self::plan_scoped(config, max_chunk, mode, &Qwen4BandScope::single(config))
    }

    /// Every field of a `max_chunk`-row scratch in [`ScratchField`] order
    /// (allocation or borrowed view, with its dtype and element count), plus
    /// the host PLE staging bytes.
    ///
    /// `Full` buffers are F32 and sized for the padded-16 slot bound of
    /// `max_chunk` rows with no WMMA dependence, which is the capacity the
    /// gfx942 qt=55 FP8 route reads in place: sorted slots padded to 16, grouped
    /// gate/up `[G, 2*moe]` and grouped down `[G, hidden]` F32, `rot_batch`
    /// `[slots, moe]` F32 and, in `gate_batch`, the tile descriptor
    /// (16 bytes per tile). `check_fp8_grouped_scratch` proves it at load.
    ///
    /// `Full` is one dedicated allocation per field.  `FnSymGfx1201` applies,
    /// on top of the same field list, with `n = max_chunk`, `c = min(n,
    /// QWEN4_COMPACT_FALLBACK_ROWS)` and `G(r)` the tile-padded grouped rows of
    /// `r` rows:
    /// - gate/up/rot batches: `c * top_k * moe_intermediate` F32;
    /// - expanded down: `max(c*top_k*hidden*4, n*hidden*4, n*top_k*8)` bytes
    ///   (the shared-down F32 fallback and combine-order bytes are reserved
    ///   regardless of dtype or flags);
    /// - grouped gate/up: `max(G(c)*2*moe*4, G(n)*moe*2)` bytes and grouped
    ///   down `max(G(c)*hidden*4, G(n)*hidden*2)` bytes, kept as F32-typed
    ///   facades of `bytes / 4` elements (the IU4 kernels read the pointers as
    ///   BF16 storage);
    /// - phase arena: `projection` owns one allocation; the step-local fields
    ///   of the GDN, QSA, HC, PLE, MoE and embedding steps (see
    ///   [`compact_arena_groups`]) are views of it, each group laid out from
    ///   element 0 with every field start rounded up to 256 bytes, and the
    ///   arena is the largest group.  `ple_gated` -> `rotation`.
    ///   `ple_normed` stays dedicated but holds only
    ///   [`QWEN4_PLE_NORMED_KEEP_ROWS`] rows (partial verify rollback reads it
    ///   after the forward); a wider forward writes `ple_normed_wide`, a PLE
    ///   group member, and every post-forward reader then refuses.
    fn plan_scoped(
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
    ) -> Result<(Vec<ScratchSlot>, usize), Qwen4GpuForwardError> {
        use ScratchField as F;
        if max_chunk == 0 {
            return Err(invalid("max_chunk is zero"));
        }
        let dims = program_dims(config);
        let layout = Qwen4ScratchLayout::for_rows(dims, max_chunk)
            .map_err(|error| invalid(error.to_string()))?;
        let hidden = layout.hidden;
        let wide = layout.wide;
        let q_width = layout.q_width;
        let qsa_qgate = 2 * q_width;
        let qsa_index = layout.index_width;
        let gdn_value = layout.gdn_value;
        let gdn_qkv = layout.gdn_qkv;
        // A band plans only the fields its operations touch (the S5
        // `forward_census_scope` terms): GDN/QSA fields for the layer kinds it
        // owns, token/embedding/PLE fields on the input band, logits on the
        // output band. An absent field is a zero-element slot: never
        // allocated, never freed, never read. The whole-model scope keeps
        // every field, so its plan is the historical one.
        let single = *scope == Qwen4BandScope::single(config);
        let has = |kind: LayerType| {
            single
                || scope
                    .band
                    .clone()
                    .any(|layer| config.layer_types.get(layer) == Some(&kind))
        };
        let (has_gdn, has_qsa) = (
            has(LayerType::LinearAttention),
            has(LayerType::FullAttention),
        );
        let max_projection = wide
            .max(hidden)
            .max(if has_qsa { qsa_qgate } else { 0 })
            .max(if has_gdn { gdn_qkv } else { 0 });
        let max_rotation = wide.max(hidden).max(config.ffn.intermediate_size());
        // PLE channels are `hc_count * hidden` (the key projection's rows),
        // not `ple_embed_dim * hc_count`; the staged row width is
        // `head_count * row_width = ple_embed_dim`.
        let ple_embed_dim = config
            .ple
            .as_ref()
            .map_or(0, |ple| ple.head_count * ple.row_width);
        let ple_channels = wide;
        let ffn_batch_rows = layout.routed_rows.max(layout.dense_rows);
        let max_ple_bytes = max_chunk
            .checked_mul(if scope.input { ple_embed_dim } else { 0 })
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| invalid("PLE staging size overflow"))?;
        let i32_bytes = std::mem::size_of::<i32>();
        let qsa_selected_bytes = max_chunk
            .checked_mul(config.qsa_selected_capacity())
            .and_then(|elements| elements.checked_mul(i32_bytes))
            .ok_or_else(|| invalid("QSA selected scratch size overflow"))?;
        let route_i32_bytes = layout
            .slots
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE route index scratch overflow"))?;
        let expert_i32_bytes = config
            .num_experts
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE expert index scratch overflow"))?;
        let expert_offsets_bytes = config
            .num_experts
            .checked_add(1)
            .and_then(|count| count.checked_mul(i32_bytes))
            .ok_or_else(|| invalid("MoE expert offset scratch overflow"))?;
        let grouped_bound = layout.grouped_rows;
        let grouped_i32_bytes = grouped_bound
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE grouped index scratch overflow"))?;
        let tile_i32_bytes = (grouped_bound / 16)
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE grouped tile scratch overflow"))?;
        let rows = |width: usize| scratch_mul(max_chunk, width);
        let routed_width = scratch_mul(config.num_experts_per_tok, config.moe_intermediate_size)?;
        let present = |field: ScratchField| match field {
            F::TokenIds
            | F::EmbeddingRot
            | F::Embeddings
            | F::PleStaged
            | F::PleRows
            | F::PleKey
            | F::PleValue
            | F::PleQuery
            | F::PleGated
            | F::PleNormed
            | F::PleOutput => scope.input,
            F::PleNormedWide => {
                scope.input
                    && mode == Qwen4ForwardScratchMode::FnSymGfx1201
                    && max_chunk > QWEN4_PLE_NORMED_KEEP_ROWS
            }
            F::GdnA
            | F::GdnB
            | F::GdnGate
            | F::GdnBeta
            | F::GdnRecurrentOutput
            | F::GdnZ
            | F::GdnOutput => has_gdn,
            F::QsaIndex | F::QsaQgate | F::QsaK | F::QsaV | F::QsaOutput | F::QsaSelected => {
                has_qsa
            }
            F::Logits => scope.output,
            _ => true,
        };
        let mut slots: Vec<ScratchSlot> = Vec::with_capacity(ScratchField::COUNT);
        let mut alloc = |field: ScratchField,
                         elements: usize,
                         dtype: DType|
         -> Result<(), Qwen4GpuForwardError> {
            if slots.len() != field as usize {
                return Err(invalid(format!(
                    "scratch plan field {field:?} is out of order"
                )));
            }
            // Every planned buffer holds at least one element: MoE-only
            // extents are zero for a dense FFN.
            slots.push(ScratchSlot {
                elements: if present(field) { elements.max(1) } else { 0 },
                dtype,
                owner: None,
                pitch: None,
                offset: 0,
            });
            Ok(())
        };
        let result = (|| {
            alloc(F::TokenIds, rows(std::mem::size_of::<i32>())?, DType::Raw)?;
            alloc(F::EmbeddingRot, rows(hidden)?, DType::F32)?;
            alloc(F::Embeddings, rows(hidden)?, DType::F32)?;
            alloc(F::Streams, rows(wide)?, DType::F32)?;
            alloc(F::HcNormalized, rows(wide)?, DType::F32)?;
            alloc(F::HcLow, rows(config.hc_lowrank)?, DType::F32)?;
            alloc(F::HcUp, rows(wide)?, DType::F32)?;
            alloc(F::HcMixed, rows(hidden)?, DType::F32)?;
            alloc(F::HcGates, rows(config.hc_count)?, DType::F32)?;
            alloc(F::Rotation, rows(max_rotation)?, DType::F32)?;
            alloc(F::Projection, rows(max_projection)?, DType::F32)?;
            alloc(F::Projection2, rows(max_projection)?, DType::F32)?;
            alloc(F::GdnA, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnB, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnGate, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnBeta, rows(config.linear_num_value_heads)?, DType::F32)?;
            alloc(F::GdnRecurrentOutput, rows(gdn_value)?, DType::F32)?;
            alloc(
                F::GdnBf16,
                (if has_gdn { gdn_value } else { 0 })
                    .max(hidden)
                    .max(config.hc_lowrank),
                DType::BF16,
            )?;
            alloc(F::GdnZ, rows(gdn_value)?, DType::F32)?;
            alloc(F::GdnOutput, rows(gdn_value)?, DType::F32)?;
            alloc(F::QsaIndex, rows(qsa_index)?, DType::F32)?;
            alloc(F::QsaQgate, rows(qsa_qgate)?, DType::F32)?;
            alloc(
                F::QsaK,
                rows(scratch_mul(config.num_key_value_heads, config.head_dim)?)?,
                DType::F32,
            )?;
            alloc(
                F::QsaV,
                rows(scratch_mul(config.num_key_value_heads, config.head_dim)?)?,
                DType::F32,
            )?;
            alloc(F::QsaOutput, rows(q_width)?, DType::F32)?;
            alloc(F::QsaSelected, qsa_selected_bytes, DType::Raw)?;
            alloc(F::AttentionOutput, rows(hidden)?, DType::F32)?;
            alloc(F::PleStaged, rows(ple_embed_dim)?, DType::BF16)?;
            alloc(F::PleRows, rows(ple_embed_dim.max(hidden))?, DType::F32)?;
            alloc(F::PleKey, rows(ple_channels)?, DType::F32)?;
            alloc(F::PleValue, rows(hidden)?, DType::F32)?;
            alloc(F::PleQuery, rows(ple_channels)?, DType::F32)?;
            alloc(F::PleGated, rows(ple_channels)?, DType::F32)?;
            // Full keeps every row of the forward; the compact plan keeps the
            // rows a verify rollback can read after the forward and routes a
            // wider forward's normed rows through `PleNormedWide`.
            alloc(
                F::PleNormed,
                if mode == Qwen4ForwardScratchMode::FnSymGfx1201 {
                    scratch_mul(max_chunk.min(QWEN4_PLE_NORMED_KEEP_ROWS), ple_channels)?
                } else {
                    rows(ple_channels)?
                },
                DType::F32,
            )?;
            alloc(F::PleNormedWide, rows(ple_channels)?, DType::F32)?;
            alloc(F::PleOutput, rows(ple_channels)?, DType::F32)?;
            // Viewed as `[rows, router_columns]` by every forward (one
            // placeholder column for a dense FFN).
            alloc(F::RouterLogits, rows(router_columns(dims))?, DType::F32)?;
            alloc(F::MoeXRot, rows(hidden)?, DType::F32)?;
            alloc(
                F::MoeGateUp,
                rows(2 * config.moe_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeGate,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeUp,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeHidden,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(F::FfnOutput, rows(hidden)?, DType::F32)?;
            alloc(F::MoeSharedOutput, rows(hidden)?, DType::F32)?;
            // Batched gate/up rows: routed slots * routed intermediate for a
            // MoE profile, rows * intermediate for a dense FFN (exactly one of
            // the two extents is nonzero); `Step::GatedMlp` borrows both.
            alloc(F::MoeGateBatch, ffn_batch_rows, DType::F32)?;
            alloc(F::MoeUpBatch, ffn_batch_rows, DType::F32)?;
            alloc(F::MoeRotBatch, rows(routed_width)?, DType::F32)?;
            alloc(F::MoeTopkIndices, route_i32_bytes, DType::Raw)?;
            alloc(
                F::MoeTopkWeights,
                rows(config.num_experts_per_tok)?,
                DType::F32,
            )?;
            alloc(
                F::MoeDownExpanded,
                rows(scratch_mul(config.num_experts_per_tok, hidden)?)?,
                DType::F32,
            )?;
            alloc(
                F::MoeScalar,
                rows(config.shared_expert_intermediate_size)?,
                DType::F32,
            )?;
            alloc(F::MoeExpertTokenCounts, expert_i32_bytes, DType::Raw)?;
            alloc(F::MoeExpertOffsets, expert_offsets_bytes, DType::Raw)?;
            alloc(F::MoeSortedSlotIndex, grouped_i32_bytes, DType::Raw)?;
            alloc(F::MoeExpertTileIds, tile_i32_bytes, DType::Raw)?;
            alloc(F::MoeInversePerm, route_i32_bytes, DType::Raw)?;
            alloc(
                F::MoeYGateUpGrouped,
                scratch_mul(grouped_bound, 2 * config.moe_intermediate_size)?,
                DType::F32,
            )?;
            alloc(
                F::MoeYDownGrouped,
                scratch_mul(grouped_bound, hidden)?,
                DType::F32,
            )?;
            alloc(F::Logits, config.vocab_size, DType::F32)?;
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        result?;
        if slots.len() != ScratchField::COUNT {
            return Err(invalid("scratch plan does not cover every field"));
        }
        if mode == Qwen4ForwardScratchMode::FnSymGfx1201 {
            if ple_channels != wide {
                return Err(invalid(
                    "compact scratch requires PLE channels equal to the hyper-connection width",
                ));
            }
            let f32_bytes = std::mem::size_of::<f32>();
            let bf16_bytes = DType::BF16.size();
            let top_k = config.num_experts_per_tok;
            let moe = config.moe_intermediate_size;
            let fallback_rows = max_chunk.min(QWEN4_COMPACT_FALLBACK_ROWS);
            let fallback_grouped = crate::program::grouped_m_total_bound(
                scratch_mul(fallback_rows, top_k)?,
                config.num_experts,
            )
            .map_err(|error| invalid(error.to_string()))?;
            // Elements of an F32-typed facade over `bytes` of storage.
            let facade = |bytes: usize| bytes.div_ceil(f32_bytes);
            let fallback_batch = scratch_mul(fallback_rows, routed_width)?;
            for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
                slots[field as usize].elements = fallback_batch;
            }
            let hidden_f32_bytes = scratch_mul(hidden, f32_bytes)?;
            let expanded_bytes = scratch_mul(scratch_mul(fallback_rows, top_k)?, hidden_f32_bytes)?
                .max(scratch_mul(max_chunk, hidden_f32_bytes)?)
                .max(scratch_mul(scratch_mul(max_chunk, top_k)?, 8)?);
            slots[F::MoeDownExpanded as usize].elements = facade(expanded_bytes);
            let gate_up_f32 = scratch_mul(fallback_grouped, scratch_mul(2 * moe, f32_bytes)?)?;
            let gate_up_bf16 = scratch_mul(grouped_bound, scratch_mul(moe, bf16_bytes)?)?;
            slots[F::MoeYGateUpGrouped as usize].elements = facade(gate_up_f32.max(gate_up_bf16));
            let down_f32 = scratch_mul(fallback_grouped, hidden_f32_bytes)?;
            let down_bf16 = scratch_mul(grouped_bound, scratch_mul(hidden, bf16_bytes)?)?;
            slots[F::MoeYDownGrouped as usize].elements = facade(down_f32.max(down_bf16));

            // Phase arena: `Projection` owns one allocation; every field of
            // a step-local group is a view of it at a 64-element-aligned
            // offset.  Groups never run at once, so they overlap.
            let groups = compact_arena_groups(config, max_chunk, &slots)?;
            let mut arena = 0usize;
            let mut placed: Vec<(F, usize, Option<usize>)> = Vec::new();
            for group in &groups {
                let mut cursor = 0usize;
                for &(field, elements, pitch) in group {
                    if slots[field as usize].elements == 0 || (field == F::Projection && !has_gdn) {
                        continue;
                    }
                    let offset = round_up_arena(cursor)?;
                    let extent = elements.max(match pitch {
                        Some(pitch) => scratch_mul(max_chunk, pitch)?,
                        None => 0,
                    });
                    cursor = offset
                        .checked_add(extent)
                        .ok_or_else(|| invalid("compact scratch arena overflows"))?;
                    placed.push((field, offset, pitch));
                    // `QsaSelected` keeps its byte `elements` (Raw); the group
                    // entry is the F32 cover of those bytes.
                    if field != F::Projection && field != F::QsaSelected {
                        slots[field as usize].elements = elements;
                    }
                }
                arena = arena.max(round_up_arena(cursor)?);
            }
            let owner_elements = arena.max(1);
            for &(field, offset, pitch) in &placed {
                if field == F::Projection {
                    continue;
                }
                let slot = &mut slots[field as usize];
                // Only `QsaSelected` is a Raw member; its arena extent is the
                // F32 cover of its bytes.
                let raw_member = field == F::QsaSelected && slot.dtype == DType::Raw;
                if slot.dtype != DType::F32 && !raw_member {
                    return Err(invalid(format!(
                        "scratch field {field:?} cannot alias {:?}",
                        F::Projection
                    )));
                }
                let slot_elements = if raw_member {
                    slot.elements.div_ceil(std::mem::size_of::<f32>())
                } else {
                    slot.elements
                };
                if offset % COMPACT_ARENA_ALIGN != 0 {
                    return Err(invalid(format!(
                        "scratch field {field:?} offset {offset} is not 256-byte aligned"
                    )));
                }
                if offset
                    .checked_add(slot_elements)
                    .is_none_or(|end| end > owner_elements)
                {
                    return Err(invalid(format!(
                        "scratch field {field:?} ({slot_elements} elements at {offset}) exceeds \
                         the {owner_elements}-element arena"
                    )));
                }
                if let Some(pitch) = pitch {
                    let reserved = scratch_mul(max_chunk, pitch)?;
                    if slot_elements > reserved
                        || offset
                            .checked_add(reserved)
                            .is_none_or(|end| end > owner_elements)
                    {
                        return Err(invalid(format!(
                            "scratch field {field:?} is wider than its pitch {pitch} or leaves \
                             the arena"
                        )));
                    }
                }
                slot.owner = Some(F::Projection);
                slot.offset = offset;
                slot.pitch = pitch;
            }
            if present(F::Projection) {
                slots[F::Projection as usize].elements = owner_elements;
                slots[F::Projection as usize].pitch = has_gdn.then_some(gdn_qkv);
            }
            // `PleGated` stays a view of `Rotation`.
            if present(F::PleGated) {
                let owner_slot = slots[F::Rotation as usize];
                let slot = &mut slots[F::PleGated as usize];
                if slot.dtype != owner_slot.dtype
                    || slot.elements > owner_slot.elements
                    || slot.elements > scratch_mul(max_chunk, max_rotation)?
                    || owner_slot.elements < scratch_mul(max_chunk, max_rotation)?
                {
                    return Err(invalid("scratch field PleGated cannot alias Rotation"));
                }
                slot.owner = Some(F::Rotation);
                slot.pitch = Some(max_rotation);
                slots[F::Rotation as usize].pitch = Some(max_rotation);
            }
        }
        Ok((slots, max_ple_bytes))
    }

    /// Device bytes [`Self::new`] allocates for `max_chunk` rows: the owning
    /// slots only; borrowed views are not charged.
    pub(crate) fn device_bytes(
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<u64, Qwen4GpuForwardError> {
        Self::device_bytes_scoped(config, max_chunk, mode, &Qwen4BandScope::single(config))
    }

    /// Elements of the `FnSymGfx1201` phase arena (the `Projection` owner) a
    /// band of `scope` plans for `max_chunk` rows; an error is a geometry the
    /// compact plan refuses.
    pub(crate) fn compact_arena_elements(
        config: &Qwen4Config,
        max_chunk: usize,
        scope: &Qwen4BandScope,
    ) -> Result<usize, Qwen4GpuForwardError> {
        let (slots, _) = Self::plan_scoped(
            config,
            max_chunk,
            Qwen4ForwardScratchMode::FnSymGfx1201,
            scope,
        )?;
        Ok(slots[ScratchField::Projection as usize].elements)
    }

    /// [`Self::device_bytes`] for one band: only the fields it plans.
    pub(crate) fn device_bytes_scoped(
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
    ) -> Result<u64, Qwen4GpuForwardError> {
        let (slots, _) = Self::plan_scoped(config, max_chunk, mode, scope)?;
        let total = Self::owning_slot_bytes(&slots).try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes?)
                .ok_or_else(|| invalid("scratch device bytes overflow"))
        });
        total
    }

    /// Bytes of each owning slot of a plan, the only slots
    /// [`Self::allocate`] hands to `gpu.zeros` (borrowed views and absent
    /// zero-element fields are not allocations).
    fn owning_slot_bytes(
        slots: &[ScratchSlot],
    ) -> impl Iterator<Item = Result<u64, Qwen4GpuForwardError>> + '_ {
        slots
            .iter()
            .filter(|slot| slot.owner.is_none())
            .map(|slot| slot.bytes().map(|bytes| bytes as u64))
            .filter(|bytes| !matches!(bytes, Ok(0)))
    }

    /// CPU-only: pack the owning roots of `plan_scoped(config, rows, mode, scope)` into one span,
    /// each root start aligned to 256 bytes, in [`ScratchField`] order. `TokenIds` is excluded
    /// (it stays a persistent owner shared by both banks: `uploaded_token_ids` outlives the
    /// forward). Borrowed alias views (`owner.is_some()`) lie inside their owner's root by
    /// construction, so the span holds every byte any view can address.
    pub(crate) fn borrowed_backing(
        config: &Qwen4Config,
        rows: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
    ) -> Result<ScratchBacking, Qwen4GpuForwardError> {
        let (slots, _) = Self::plan_scoped(config, rows, mode, scope)?;
        let overflow = || invalid("borrowed scratch backing overflows");
        let align = |x: u64| -> Result<u64, Qwen4GpuForwardError> {
            x.checked_add(DECODE_SHARE_ROOT_ALIGN - 1)
                .map(|x| x & !(DECODE_SHARE_ROOT_ALIGN - 1))
                .ok_or_else(overflow)
        };
        let mut roots = Vec::with_capacity(slots.len());
        let mut cursor = 0u64;
        for (index, slot) in slots.iter().enumerate() {
            let field = ScratchField::ALL[index];
            if field == ScratchField::TokenIds || slot.owner.is_some() {
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

    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        Self::new_scoped(
            gpu,
            config,
            max_chunk,
            mode,
            &Qwen4BandScope::single(config),
        )
    }

    /// [`Self::new`] for one pipeline-parallel band: allocates only the
    /// fields [`Self::plan_scoped`] plans for `scope`.
    pub fn new_scoped(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        Self::allocate(gpu, config, max_chunk, mode, None, scope)
    }

    /// Allocate only state live across the A/B switch. All borrowed pointers
    /// remain owned by `bank_a`, which must outlive this bank and GPU work.
    fn new_npu_bank(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        bank_a: &Self,
        emulated: bool,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        if bank_a.mode != Qwen4ForwardScratchMode::Full {
            return Err(invalid("NPU shared bank requires full scratch"));
        }
        let kind = if emulated {
            NpuBankKind::Emulated
        } else {
            NpuBankKind::Npu
        };
        Self::allocate(
            gpu,
            config,
            bank_a.max_chunk,
            bank_a.mode,
            Some((bank_a, kind)),
            &Qwen4BandScope::single(config),
        )
    }

    #[cfg(test)]
    fn npu_bank_device_bytes(
        config: &Qwen4Config,
        max_chunk: usize,
    ) -> Result<u64, Qwen4GpuForwardError> {
        let (slots, _) = Self::plan(config, max_chunk, Qwen4ForwardScratchMode::Full)?;
        Self::npu_bank_plan_bytes(&slots, false)
    }

    fn npu_bank_plan_bytes(
        slots: &[ScratchSlot],
        emulated: bool,
    ) -> Result<u64, Qwen4GpuForwardError> {
        let kind = if emulated {
            NpuBankKind::Emulated
        } else {
            NpuBankKind::Npu
        };
        slots
            .iter()
            .enumerate()
            .try_fold(0u64, |total, (index, slot)| {
                // The exhaustive field table is shared by allocation and freeing.
                let private = kind.owns(ScratchField::ALL[index]);
                total
                    .checked_add(if private { slot.bytes()? as u64 } else { 0 })
                    .ok_or_else(|| invalid("NPU bank device bytes overflow"))
            })
    }

    /// Bind every view slot of `slots` to its owner's tensor and every absent
    /// (zero-element) field to a zero-length placeholder. `tensors` holds the
    /// owning roots (allocated or placed in a borrowed span) at their field
    /// indexes; the one routine both [`Self::allocate_with`] and
    /// [`Self::new_borrowed`] build views with, so a borrowed bank's offsets,
    /// pitches and dtypes are the allocated bank's by construction.
    fn bind_views(slots: &[ScratchSlot], tensors: &mut [Option<GpuTensor>]) {
        for (index, slot) in slots.iter().enumerate() {
            if let Some(owner) = slot.owner {
                let owner_tensor = tensors[owner as usize]
                    .as_ref()
                    .expect("plan validated that a view's owner owns its allocation");
                let view = if slot.dtype == owner_tensor.dtype {
                    owner_tensor.sub_offset(slot.offset, slot.elements)
                } else {
                    // A Raw member (`QsaSelected`): an F32-covered region
                    // retyped to its byte length.
                    let mut view = owner_tensor.sub_offset(
                        slot.offset,
                        slot.elements.div_ceil(owner_tensor.dtype.size()),
                    );
                    view.shape = vec![slot.elements];
                    view.dtype = slot.dtype;
                    view
                };
                tensors[index] = Some(view);
            }
        }
        // Absent band fields: zero-length borrowed views (never dereferenced,
        // never freed) so every field stays bound; every use is pin-gated.
        for (index, slot) in slots.iter().enumerate() {
            if slot.elements == 0 {
                let mut placeholder = tensors[ScratchField::Streams as usize]
                    .as_ref()
                    .expect("the HC streams are owned by every band")
                    .sub_offset(0, 0);
                placeholder.dtype = slot.dtype;
                tensors[index] = Some(placeholder);
            }
        }
    }

    /// `TokenIds` of a decode-share bank is a view of the persistent
    /// `token_ids` (`Raw`, the prompt chunk's rows), shared by every bank of
    /// the load because `uploaded_token_ids` outlives a prefill forward.
    fn check_external_ids(
        slots: &[ScratchSlot],
        token_ids: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        let slot = &slots[ScratchField::TokenIds as usize];
        if token_ids.dtype != slot.dtype || token_ids.numel() < slot.elements {
            return Err(invalid(format!(
                "persistent token ids ({:?} x {}) cannot back a bank's {:?} x {} token ids",
                token_ids.dtype,
                token_ids.numel(),
                slot.dtype,
                slot.elements
            )));
        }
        Ok(())
    }

    /// The resident bank of a decode-share load: [`Self::new_scoped`] for `rows`
    /// rows, except that `TokenIds` aliases the persistent `token_ids` (which
    /// holds the prompt chunk's rows and is freed once by its owner), so the
    /// bank allocates and frees no token-id storage.
    pub(crate) fn new_resident_scoped(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        rows: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
        token_ids: &GpuTensor,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        Self::allocate_with(gpu, config, rows, mode, None, scope, Some(token_ids))
    }

    /// A prefill bank of a decode-share load for `rows` rows: no allocation, no
    /// zeroing. Every owning root of `plan_scoped(config, rows, mode, scope)`
    /// is a non-owning tensor at `base + root.offset` of the external span
    /// ([`Self::borrowed_backing`] places them; the span is `span_bytes` long),
    /// every view slot is built from its owner by [`Self::bind_views`] exactly
    /// as [`Self::allocate_with`] builds it, and `TokenIds` is a view of the
    /// persistent `token_ids`. Host vectors are returned as `allocate` returns
    /// them (sized for this bank); the caller keeps the largest bank's.
    /// [`Self::free_gpu`] frees nothing of a borrowed bank.
    pub(crate) fn new_borrowed(
        config: &Qwen4Config,
        rows: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
        base: usize,
        span_bytes: usize,
        token_ids: &GpuTensor,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        let (slots, max_ple_bytes) = Self::plan_scoped(config, rows, mode, scope)?;
        let backing = Self::borrowed_backing(config, rows, mode, scope)?;
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
        Self::check_external_ids(&slots, token_ids)?;
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
            // SAFETY: `root.offset + root.bytes <= backing.span_bytes <=
            // span_bytes`, and the caller guarantees `[base, base + span_bytes)`
            // is live device memory for the bank's life. The tensor is a
            // `Borrowed` buffer: no free path ever receives it
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
        let ids_slot = &slots[ScratchField::TokenIds as usize];
        tensors[ScratchField::TokenIds as usize] = Some(token_ids.sub_offset(0, ids_slot.elements));
        for (index, slot) in slots.iter().enumerate() {
            if slot.owner.is_none() && slot.elements != 0 && tensors[index].is_none() {
                return Err(invalid(format!(
                    "borrowed scratch backing places no root for {:?}",
                    ScratchField::ALL[index]
                )));
            }
        }
        Self::bind_views(&slots, &mut tensors);
        Ok(Self::from_tensors(
            rows,
            mode,
            None,
            ScratchBackingKind::Borrowed,
            slots,
            tensors,
            max_ple_bytes,
        ))
    }

    fn allocate(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        bank: Option<(&Self, NpuBankKind)>,
        scope: &Qwen4BandScope,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        Self::allocate_with(gpu, config, max_chunk, mode, bank, scope, None)
    }

    /// [`Self::allocate`]; with `external_ids` the bank's `TokenIds` is a
    /// non-owning alias of that persistent tensor instead of an allocation.
    fn allocate_with(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        bank: Option<(&Self, NpuBankKind)>,
        scope: &Qwen4BandScope,
        external_ids: Option<&GpuTensor>,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        let bank_a = bank.map(|(bank_a, _)| bank_a);
        let bank_kind = bank.map(|(_, kind)| kind);
        let (slots, max_ple_bytes) = Self::plan_scoped(config, max_chunk, mode, scope)?;
        if let Some(token_ids) = external_ids {
            Self::check_external_ids(&slots, token_ids)?;
        }
        // Owners first; a view needs its owner's pointer.
        let mut tensors: Vec<Option<GpuTensor>> = Vec::with_capacity(slots.len());
        for (index, slot) in slots.iter().enumerate() {
            if slot.owner.is_some()
                || slot.elements == 0
                || bank_kind.is_some_and(|kind| !kind.owns(ScratchField::ALL[index]))
                || (external_ids.is_some() && ScratchField::ALL[index] == ScratchField::TokenIds)
            {
                tensors.push(None);
                continue;
            }
            match gpu.zeros(&[slot.elements], slot.dtype) {
                Ok(tensor) => tensors.push(Some(tensor)),
                Err(error) => {
                    // Only owners are present: each is freed exactly once.
                    for tensor in tensors.into_iter().flatten() {
                        let _ = gpu.free_tensor(tensor);
                    }
                    return Err(error.into());
                }
            }
        }
        if let Some(token_ids) = external_ids {
            // The whole persistent tensor (the prompt chunk's rows): the ids a
            // prefill forward uploaded are read through this bank afterwards
            // (`uploaded_token_ids`, the MTP head append).
            tensors[ScratchField::TokenIds as usize] =
                Some(token_ids.sub_offset(0, token_ids.numel()));
        }
        Self::bind_views(&slots, &mut tensors);
        if let Some(bank_a) = bank_a {
            for (index, slot) in slots.iter().enumerate() {
                let field = ScratchField::ALL[index];
                if !bank_kind.expect("a borrowed bank has a kind").owns(field) {
                    tensors[index] = Some(bank_a.field_tensor(field).sub_offset(0, slot.elements));
                }
            }
        }
        let backing = if external_ids.is_some() {
            ScratchBackingKind::ExternalIds
        } else {
            ScratchBackingKind::Owned
        };
        Ok(Self::from_tensors(
            max_chunk,
            mode,
            bank_kind,
            backing,
            slots,
            tensors,
            max_ple_bytes,
        ))
    }

    /// Assemble a bank from its bound tensors (every field exactly once) and
    /// the host vectors an allocation returns: `max_chunk` token ids and the
    /// PLE staging bytes.
    fn from_tensors(
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        bank_kind: Option<NpuBankKind>,
        backing: ScratchBackingKind,
        slots: Vec<ScratchSlot>,
        mut tensors: Vec<Option<GpuTensor>>,
        max_ple_bytes: usize,
    ) -> (Self, Vec<u8>, Vec<u8>) {
        let mut take = |field: ScratchField| {
            tensors[field as usize]
                .take()
                .expect("every scratch field is bound exactly once")
        };
        use ScratchField as F;
        let scratch = Self {
            max_chunk,
            mode,
            bank_kind,
            backing,
            token_ids: take(F::TokenIds),
            embedding_rot: take(F::EmbeddingRot),
            embeddings: take(F::Embeddings),
            streams: take(F::Streams),
            hc_normalized: take(F::HcNormalized),
            hc_low: take(F::HcLow),
            hc_up: take(F::HcUp),
            hc_mixed: take(F::HcMixed),
            hc_gates: take(F::HcGates),
            rotation: take(F::Rotation),
            projection: take(F::Projection),
            projection2: take(F::Projection2),
            gdn_a: take(F::GdnA),
            gdn_b: take(F::GdnB),
            gdn_gate: take(F::GdnGate),
            gdn_beta: take(F::GdnBeta),
            gdn_recurrent_output: take(F::GdnRecurrentOutput),
            gdn_bf16: take(F::GdnBf16),
            gdn_z: take(F::GdnZ),
            gdn_output: take(F::GdnOutput),
            qsa_index: take(F::QsaIndex),
            qsa_qgate: take(F::QsaQgate),
            qsa_k: take(F::QsaK),
            qsa_v: take(F::QsaV),
            qsa_output: take(F::QsaOutput),
            qsa_selected: take(F::QsaSelected),
            attention_output: take(F::AttentionOutput),
            ple_staged: take(F::PleStaged),
            ple_rows: take(F::PleRows),
            ple_key: take(F::PleKey),
            ple_value: take(F::PleValue),
            ple_query: take(F::PleQuery),
            ple_gated: take(F::PleGated),
            ple_normed: take(F::PleNormed),
            ple_normed_wide: take(F::PleNormedWide),
            ple_normed_in_wide: std::sync::atomic::AtomicBool::new(false),
            ple_output: take(F::PleOutput),
            router_logits: take(F::RouterLogits),
            moe_x_rot: take(F::MoeXRot),
            moe_gate_up: take(F::MoeGateUp),
            moe_gate: take(F::MoeGate),
            moe_up: take(F::MoeUp),
            moe_hidden: take(F::MoeHidden),
            ffn_output: take(F::FfnOutput),
            moe_shared_output: take(F::MoeSharedOutput),
            moe_gate_batch: take(F::MoeGateBatch),
            moe_up_batch: take(F::MoeUpBatch),
            moe_rot_batch: take(F::MoeRotBatch),
            moe_topk_indices: take(F::MoeTopkIndices),
            moe_topk_weights: take(F::MoeTopkWeights),
            moe_down_expanded: take(F::MoeDownExpanded),
            moe_scalar: take(F::MoeScalar),
            moe_expert_token_counts: take(F::MoeExpertTokenCounts),
            moe_expert_offsets: take(F::MoeExpertOffsets),
            moe_sorted_slot_index: take(F::MoeSortedSlotIndex),
            moe_expert_tile_ids: take(F::MoeExpertTileIds),
            moe_inverse_perm: take(F::MoeInversePerm),
            moe_y_gate_up_grouped: take(F::MoeYGateUpGrouped),
            moe_y_down_grouped: take(F::MoeYDownGrouped),
            logits: take(F::Logits),
            slots,
        };
        (
            scratch,
            vec![0; max_chunk * std::mem::size_of::<i32>()],
            vec![0; max_ple_bytes],
        )
    }

    /// The tensor of `field`: an owner's allocation or a borrowed view.
    pub(crate) fn field_tensor(&self, field: ScratchField) -> &GpuTensor {
        use ScratchField as F;
        match field {
            F::TokenIds => &self.token_ids,
            F::EmbeddingRot => &self.embedding_rot,
            F::Embeddings => &self.embeddings,
            F::Streams => &self.streams,
            F::HcNormalized => &self.hc_normalized,
            F::HcLow => &self.hc_low,
            F::HcUp => &self.hc_up,
            F::HcMixed => &self.hc_mixed,
            F::HcGates => &self.hc_gates,
            F::Rotation => &self.rotation,
            F::Projection => &self.projection,
            F::Projection2 => &self.projection2,
            F::GdnA => &self.gdn_a,
            F::GdnB => &self.gdn_b,
            F::GdnGate => &self.gdn_gate,
            F::GdnBeta => &self.gdn_beta,
            F::GdnRecurrentOutput => &self.gdn_recurrent_output,
            F::GdnBf16 => &self.gdn_bf16,
            F::GdnZ => &self.gdn_z,
            F::GdnOutput => &self.gdn_output,
            F::QsaIndex => &self.qsa_index,
            F::QsaQgate => &self.qsa_qgate,
            F::QsaK => &self.qsa_k,
            F::QsaV => &self.qsa_v,
            F::QsaOutput => &self.qsa_output,
            F::QsaSelected => &self.qsa_selected,
            F::AttentionOutput => &self.attention_output,
            F::PleStaged => &self.ple_staged,
            F::PleRows => &self.ple_rows,
            F::PleKey => &self.ple_key,
            F::PleValue => &self.ple_value,
            F::PleQuery => &self.ple_query,
            F::PleGated => &self.ple_gated,
            F::PleNormed => &self.ple_normed,
            F::PleNormedWide => &self.ple_normed_wide,
            F::PleOutput => &self.ple_output,
            F::RouterLogits => &self.router_logits,
            F::MoeXRot => &self.moe_x_rot,
            F::MoeGateUp => &self.moe_gate_up,
            F::MoeGate => &self.moe_gate,
            F::MoeUp => &self.moe_up,
            F::MoeHidden => &self.moe_hidden,
            F::FfnOutput => &self.ffn_output,
            F::MoeSharedOutput => &self.moe_shared_output,
            F::MoeGateBatch => &self.moe_gate_batch,
            F::MoeUpBatch => &self.moe_up_batch,
            F::MoeRotBatch => &self.moe_rot_batch,
            F::MoeTopkIndices => &self.moe_topk_indices,
            F::MoeTopkWeights => &self.moe_topk_weights,
            F::MoeDownExpanded => &self.moe_down_expanded,
            F::MoeScalar => &self.moe_scalar,
            F::MoeExpertTokenCounts => &self.moe_expert_token_counts,
            F::MoeExpertOffsets => &self.moe_expert_offsets,
            F::MoeSortedSlotIndex => &self.moe_sorted_slot_index,
            F::MoeExpertTileIds => &self.moe_expert_tile_ids,
            F::MoeInversePerm => &self.moe_inverse_perm,
            F::MoeYGateUpGrouped => &self.moe_y_gate_up_grouped,
            F::MoeYDownGrouped => &self.moe_y_down_grouped,
            F::Logits => &self.logits,
        }
    }

    /// Rows `[begin, begin + rows)` of `field` for one lane segment, each
    /// row `logical_width` live elements wide.
    ///
    /// A dedicated field (no pitch) is addressed contiguously at its logical
    /// width from row `begin` to its end.  An alias-set member is addressed at
    /// its allocation owner's row pitch ([`owner_pitch_extent`]) and the view
    /// is the segment's whole `rows * pitch` reservation, so every member of a
    /// set maps one segment to the same rows of one owner.  The returned view
    /// is an ordinary contiguous tensor: kernels get no strided ABI.
    pub(crate) fn segment_view(
        &self,
        field: ScratchField,
        begin: usize,
        rows: usize,
        logical_width: usize,
    ) -> Result<GpuTensor, Qwen4GpuForwardError> {
        let tensor = self.field_tensor(field);
        let slot = self.slots[field as usize];
        match slot.pitch {
            None => {
                let numel = tensor.numel();
                let offset = scratch_mul(begin, logical_width)?;
                if offset > numel {
                    return Err(invalid("lane row view starts past its scratch tensor"));
                }
                Ok(tensor.sub_offset(offset, numel - offset))
            }
            Some(pitch) => {
                // The pitched reservation is bounded by the allocation that
                // actually holds it: a borrowed view's own logical element
                // count can be narrower than `rows * pitch`.
                let region = scratch_mul(self.max_chunk, pitch)?;
                let storage = self
                    .field_tensor(slot.owner.unwrap_or(field))
                    .sub_offset(slot.offset, region);
                let (offset, len) =
                    owner_pitch_extent(pitch, storage.numel(), begin, rows, logical_width)?;
                Ok(storage.sub_offset(offset, len))
            }
        }
    }

    /// The field a `rows`-row forward writes its PLE normed rows to, and
    /// whether that is the wide arena view.
    pub(crate) fn ple_normed_field(&self, rows: usize) -> ScratchField {
        if self.slots[ScratchField::PleNormedWide as usize].elements != 0
            && rows > QWEN4_PLE_NORMED_KEEP_ROWS
        {
            ScratchField::PleNormedWide
        } else {
            ScratchField::PleNormed
        }
    }

    /// The tensor a `rows`-row PLE step writes its normed rows to; a row count
    /// past [`QWEN4_PLE_NORMED_KEEP_ROWS`] takes the arena view and marks the
    /// kept rows stale.
    pub(crate) fn ple_normed_for(&self, rows: usize) -> &GpuTensor {
        let field = self.ple_normed_field(rows);
        if field == ScratchField::PleNormedWide {
            self.ple_normed_in_wide
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.field_tensor(field)
    }

    /// A forward of `rows` rows begins: the kept normed rows are valid again
    /// unless this forward is wider than them.
    pub(crate) fn note_forward_rows(&self, rows: usize) {
        self.ple_normed_in_wide.store(
            self.ple_normed_field(rows) == ScratchField::PleNormedWide,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// The kept PLE normed rows for a verify rollback; an error when the last
    /// forward was wider than [`QWEN4_PLE_NORMED_KEEP_ROWS`] and wrote its
    /// normed rows elsewhere.
    pub(crate) fn ple_normed_for_rollback(&self) -> Result<&GpuTensor, Qwen4GpuForwardError> {
        if self
            .ple_normed_in_wide
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err(invalid(format!(
                "the last forward was wider than {QWEN4_PLE_NORMED_KEEP_ROWS} rows: its PLE \
                 convolution input rows are not kept for a verify rollback"
            )));
        }
        Ok(&self.ple_normed)
    }

    /// Free the owners.  Borrowed views are never freed: their allocation is
    /// freed once, through its owner.
    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let slots = self.slots;
        let bank_kind = self.bank_kind;
        let backing = self.backing;
        let tensors = [
            self.token_ids,
            self.embedding_rot,
            self.embeddings,
            self.streams,
            self.hc_normalized,
            self.hc_low,
            self.hc_up,
            self.hc_mixed,
            self.hc_gates,
            self.rotation,
            self.projection,
            self.projection2,
            self.gdn_a,
            self.gdn_b,
            self.gdn_gate,
            self.gdn_beta,
            self.gdn_recurrent_output,
            self.gdn_bf16,
            self.gdn_z,
            self.gdn_output,
            self.qsa_index,
            self.qsa_qgate,
            self.qsa_k,
            self.qsa_v,
            self.qsa_output,
            self.qsa_selected,
            self.attention_output,
            self.ple_staged,
            self.ple_rows,
            self.ple_key,
            self.ple_value,
            self.ple_query,
            self.ple_gated,
            self.ple_normed,
            self.ple_normed_wide,
            self.ple_output,
            self.router_logits,
            self.moe_x_rot,
            self.moe_gate_up,
            self.moe_gate,
            self.moe_up,
            self.moe_hidden,
            self.ffn_output,
            self.moe_shared_output,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
            self.moe_scalar,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_expert_tile_ids,
            self.moe_inverse_perm,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
            self.logits,
        ];
        debug_assert_eq!(tensors.len(), slots.len());
        let mut first = None;
        for (index, (tensor, slot)) in tensors.into_iter().zip(slots).enumerate() {
            if !backing.frees(ScratchField::ALL[index], &slot, bank_kind) {
                continue;
            }
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

pub(crate) fn layer_scratch<'a>(
    scratch: &'a Qwen4GpuForwardScratch,
    router_logits: &'a GpuTensor,
) -> Qwen4LayerScratch<'a> {
    Qwen4LayerScratch {
        streams: &scratch.streams,
        rotation: &scratch.rotation,
        hc_normalized: &scratch.hc_normalized,
        hc_low: &scratch.hc_low,
        hc_up: &scratch.hc_up,
        hc_mixed: &scratch.hc_mixed,
        hc_gates: &scratch.hc_gates,
        projection: &scratch.projection,
        projection2: &scratch.projection2,
        gdn_a: &scratch.gdn_a,
        gdn_b: &scratch.gdn_b,
        gdn_gate: &scratch.gdn_gate,
        gdn_beta: &scratch.gdn_beta,
        gdn_recurrent_output: &scratch.gdn_recurrent_output,
        gdn_z: &scratch.gdn_z,
        gdn_output: &scratch.gdn_output,
        qsa_index: &scratch.qsa_index,
        qsa_qgate: &scratch.qsa_qgate,
        qsa_k: &scratch.qsa_k,
        qsa_v: &scratch.qsa_v,
        qsa_output: &scratch.qsa_output,
        attention_output: &scratch.attention_output,
        ffn_output: &scratch.ffn_output,
        moe_shared_output: &scratch.moe_shared_output,
        moe_router_logits: router_logits,
        moe_x_rot: &scratch.moe_x_rot,
        moe_gate_up: &scratch.moe_gate_up,
        moe_scalar: &scratch.moe_scalar,
        moe_gate: &scratch.moe_gate,
        moe_up: &scratch.moe_up,
        moe_hidden: &scratch.moe_hidden,
        moe_gate_batch: &scratch.moe_gate_batch,
        moe_up_batch: &scratch.moe_up_batch,
        moe_rot_batch: &scratch.moe_rot_batch,
        moe_topk_indices: &scratch.moe_topk_indices,
        moe_topk_weights: &scratch.moe_topk_weights,
        moe_down_expanded: &scratch.moe_down_expanded,
        moe_expert_token_counts: &scratch.moe_expert_token_counts,
        moe_expert_offsets: &scratch.moe_expert_offsets,
        moe_sorted_slot_index: &scratch.moe_sorted_slot_index,
        moe_expert_tile_ids: &scratch.moe_expert_tile_ids,
        moe_inverse_perm: &scratch.moe_inverse_perm,
        moe_y_gate_up_grouped: &scratch.moe_y_gate_up_grouped,
        moe_y_down_grouped: &scratch.moe_y_down_grouped,
    }
}

/// The Gpu scope values [`Qwen4GpuForward::enter_tile_scopes`] replaced.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TileScopes {
    scope: bool,
    invariant: bool,
    f16_moe: bool,
    ragged: bool,
}

/// How a prefill chunk's staged PLE rows reach the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PleUploadMode {
    /// After a device-token readback the host already synchronized past the
    /// previous upload.
    Unfenced,
    /// Fenced asynchronous upload: the staging bytes stay unwritable until the
    /// copy ran, the host never waits on the device queue.
    Async,
    /// Blocking upload (capture aware).
    Blocking,
}

/// Wait for a fetched PLE row lease, stage it in `host`, upload it to
/// `staged` and widen it into `ple_rows` (F16 when `rows_f16`). The one body
/// of the PLE row staging: the chunk-major forward and every S1 layer-major
/// tile run it, so the rows reaching the PLE step are produced identically.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_ple_rows(
    gpu: &mut Gpu,
    host: &mut PleHostStage,
    ple: &mut RowFetch<'_>,
    staged: &GpuTensor,
    ple_rows: &GpuTensor,
    rows_f16: bool,
    rows: usize,
    head_count: usize,
    row_width: usize,
    mode: PleUploadMode,
) -> Result<(), Qwen4GpuForwardError> {
    let wait_started = qwen4_profile_start();
    let lease_result = ple.wait();
    qwen4_profile_record(Qwen4ProfilePhase::PleWait, wait_started);
    let lease = lease_result.map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
    let upload_len = lease
        .as_bytes()
        .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?
        .len();
    // Retire any earlier asynchronous upload still reading the staging bytes
    // before the host copy overwrites them.
    let stage_started = qwen4_profile_start();
    let stage_result = match host.stage_buffer(gpu, upload_len) {
        Ok(buffer) => lease
            .stage_into(buffer)
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string())),
        Err(error) => Err(error.into()),
    };
    qwen4_profile_record(Qwen4ProfilePhase::PleStage, stage_started);
    stage_result?;
    let upload_started = qwen4_profile_start();
    let upload_result = match mode {
        PleUploadMode::Unfenced => host.upload_unfenced(gpu, &staged.buf, upload_len),
        PleUploadMode::Async => host.upload_async(gpu, &staged.buf, upload_len),
        PleUploadMode::Blocking => host.upload_blocking(gpu, &staged.buf, upload_len),
    };
    qwen4_profile_record(Qwen4ProfilePhase::PleUpload, upload_started);
    upload_result?;
    lease
        .validate_after_upload()
        .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
    let apply_started = qwen4_profile_start();
    if rows_f16 {
        gpu.grouped_gather_convert_bf16_f16(staged, ple_rows, rows, head_count, row_width)?;
    } else {
        gpu.grouped_gather_convert_bf16(staged, ple_rows, rows, head_count, row_width)?;
    }
    qwen4_profile_record(Qwen4ProfilePhase::PleApply, apply_started);
    Ok(())
}

/// Typed per-layer step construction shared by the chunk-major forward
/// (`forward_chunk_scoped`) and the S1 layer-major prompt schedule
/// (`crate::slowlink_prefill`). Every lowering of a layer's HC read / mixer /
/// HC write / dense FFN into [`Step`]s lives in these methods; callers choose
/// only the tensors (`streams`, `gates`, the MoE output) and row geometry.
///
/// Borrowed views only: the step list may not outlive `'a`, and building one
/// never touches the device.
pub(crate) struct LayerPhaseEnv<'a> {
    pub(crate) scratch: &'a Qwen4GpuForwardScratch,
    pub(crate) decode_q8: &'a [GpuTensor],
    pub(crate) trunk_zba: &'a [Option<GpuTensor>],
    pub(crate) trunk_a4: &'a [u16],
    pub(crate) prefill_scope: bool,
    pub(crate) bundle: &'a Qwen4Bundle,
    pub(crate) dims: Qwen4ProgramDims,
    /// Decided once per program: every op touching the HC streams agrees.
    pub(crate) bf16_state: bool,
    /// The routing row count every row-count decision of the program sees.
    pub(crate) route_n: usize,
    pub(crate) few_row_limit: usize,
    /// Launch rows of this program (a tile, for S1).
    pub(crate) rows: usize,
    /// Absolute position of the program's first row (the GDN recurrence base).
    pub(crate) start_position: usize,
    /// The HC stream tensor the layer reads and writes: `scratch.streams`, or
    /// a row view of the S1 prompt arena.
    pub(crate) streams: &'a GpuTensor,
    /// The HC gate tensor of both hyper writes of the layer.
    pub(crate) gates: &'a GpuTensor,
}

impl<'a> LayerPhaseEnv<'a> {
    /// The PLE depthwise step of the PLE layer and whether it stages its rows
    /// as F16 (decided once; the staging closure agrees on it).
    pub(crate) fn ple_step(
        &self,
        gpu: &Gpu,
        ple_weights: &crate::weights::PleWeights,
        ple_head_count: usize,
        ple_row_width: usize,
    ) -> Result<(Step<'a>, bool), Qwen4GpuForwardError> {
        let (scratch, bundle) = (self.scratch, self.bundle);
        let config = &bundle.config;
        let ple = ple_desc(&bundle.weights, ple_weights)?;
        let rows_f16 = qwen4_ple_rows_f16(
            gpu,
            self.bf16_state,
            self.dims.hc_count,
            config.hidden_size,
            &ple,
            self.rows,
        );
        let step = Step::GroupedDepthwise(GroupedDepthwiseOp {
            rotation: &scratch.rotation,
            state_bf16: self.bf16_state,
            rows_f16,
            key: ple.key,
            value: ple.value,
            norm_key: ple.norm_key,
            norm_query: ple.norm_query,
            norm_conv: ple.norm_conv,
            conv: ple.conv,
            state: bundle
                .state
                .ple_conv
                .as_ref()
                .ok_or_else(|| invalid("PLE conv state missing"))?,
            streams: self.streams,
            rows_tensor: &scratch.ple_rows,
            query: &scratch.ple_query,
            key_scratch: &scratch.ple_key,
            value_scratch: &scratch.ple_value,
            gated: &scratch.ple_gated,
            normed: scratch.ple_normed_for(self.rows),
            output: &scratch.ple_output,
            rows: self.rows,
            branches: self.dims.hc_count,
            hidden: config.hidden_size,
            embed_dim: ple_head_count * ple_row_width,
            kernel_size: self.dims.ple_conv_kernel_dim,
            dilation: config.ple_conv_dilation(),
            epsilon: EPSILON,
        });
        Ok((step, rows_f16))
    }

    /// The layer's prefix: attention HC read, the mixer (GDN or QSA), the
    /// attention HC write and the MLP HC read. After it the prepared MLP input
    /// is in `scratch.hc_mixed`.
    pub(crate) fn push_mixer_prefix(
        &self,
        steps: &mut Vec<Step<'a>>,
        layer_index: usize,
        layer: &Qwen4LayerWeights,
        description: &Qwen4LayerDescription<'a>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let (scratch, bundle, dims) = (self.scratch, self.bundle, self.dims);
        let config = &bundle.config;
        let (n, route_n, few_row_limit) = (self.rows, self.route_n, self.few_row_limit);
        let bf16_state = self.bf16_state;
        let attn_read = &description.attn_hyper.read;
        if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
            return Err(invalid(
                "Qwen4 generic layer program inline capacity exhausted",
            ));
        }
        steps.push(Step::HyperRead(HyperReadOp {
            rotation: &scratch.rotation,
            state_bf16: bf16_state,
            input: self.streams,
            norm_weight: attn_read.norm,
            input_mix_down: decode_copy(
                self.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER,
                route_n,
                few_row_limit,
                attn_read.input_mix_down,
            ),
            input_mix_up: decode_copy(
                self.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER + 1,
                route_n,
                few_row_limit,
                attn_read.input_mix_up,
            ),
            normalized: &scratch.hc_normalized,
            low: &scratch.hc_low,
            up: &scratch.hc_up,
            mixed: &scratch.hc_mixed,
            rows: n,
            branches: dims.hc_count,
            hidden: config.hidden_size,
            low_rank: dims.hc_lowrank,
        }));

        match layer.kind {
            LayerType::LinearAttention => {
                let gdn_slot = bundle
                    .state
                    .gdn_local(layer_index)
                    .ok_or_else(|| invalid("GDN state slot missing"))?;
                let state = &bundle.state.gdn[gdn_slot];
                let weights = match &description.attention {
                    Qwen4AttentionWeights::Linear(weights) => weights,
                    _ => return Err(invalid("GDN descriptor kind mismatch")),
                };
                steps.push(Step::GatedDeltaNet(GatedDeltaNetOp {
                    rotation: &scratch.rotation,
                    qkv: weights.qkv,
                    conv: weights.conv,
                    in_proj_a: weights.in_proj_a,
                    in_proj_b: weights.in_proj_b,
                    a_log: weights.a_log,
                    dt_bias: weights.dt_bias,
                    z: weights.z,
                    norm: weights.norm,
                    output: weights.output,
                    recurrent: &state.recurrent,
                    format: bundle.state.gdn_format(),
                    residual: state.ef_residual.as_ref(),
                    conv_state: &state.conv,
                    projection: &scratch.projection,
                    projection2: &scratch.projection2,
                    a: &scratch.gdn_a,
                    b: &scratch.gdn_b,
                    gate: &scratch.gdn_gate,
                    beta: &scratch.gdn_beta,
                    recurrent_output: &scratch.gdn_recurrent_output,
                    bf16_scratch: &scratch.gdn_bf16,
                    z_output: &scratch.gdn_z,
                    output_scratch: &scratch.gdn_output,
                    input: &scratch.hc_mixed,
                    output_tensor: &scratch.attention_output,
                    rows: n,
                    start_position: self.start_position,
                    key_heads: dims.linear_num_key_heads,
                    value_heads: dims.linear_num_value_heads,
                    key_dim: dims.linear_key_head_dim,
                    value_dim: dims.linear_value_head_dim,
                    conv_kernel: dims.linear_conv_kernel_dim,
                    input_width: dims.hidden,
                    row_capture: bundle.state.gdn_row_capture(gdn_slot, n),
                    zba_fold: self.trunk_zba.get(layer_index).and_then(Option::as_ref),
                    trunk_a4: self.trunk_a4.get(layer_index).copied().unwrap_or(0),
                    phase: MixerPhase::Whole,
                }));
            }
            LayerType::FullAttention => {
                let state = bundle
                    .state
                    .qsa_by_layer(layer_index)
                    .ok_or_else(|| invalid("QSA state slot missing"))?;
                let weights = match &description.attention {
                    Qwen4AttentionWeights::Full(weights) => weights,
                    _ => return Err(invalid("QSA descriptor kind mismatch")),
                };
                steps.push(Step::IndexedAttention(IndexedAttentionOp {
                    rotation: &scratch.rotation,
                    mode: IndexedAttentionMode::Full,
                    indexer_qk: weights.indexer_qk,
                    indexer_q_norm: weights.indexer_q_norm,
                    indexer_k_norm: weights.indexer_k_norm,
                    q: weights.q,
                    k: weights.k,
                    v: weights.v,
                    q_norm: weights.q_norm,
                    k_norm: weights.k_norm,
                    output: weights.output,
                    state: IndexedAttentionState {
                        format: state.format,
                        full_keys: &state.full_keys,
                        full_values: &state.full_values,
                        raw_index_keys: &state.raw_index_keys,
                        pooled_keys: &state.pooled_keys,
                        selected_indices: &state.selected_indices,
                        // Decode and speculative verify never consume the arena.
                        f16_arena: if self.prefill_scope {
                            state.f16_arena_ref()
                        } else {
                            None
                        },
                        // The pooled BF16 mirror feeds decode and prefill scoring alike.
                        pooled_bf16: state.pooled_bf16_ref(),
                        full_capacity: state.full_capacity,
                        raw_capacity: state.raw_capacity,
                        pooled_capacity: state.pooled_capacity,
                        selected_capacity: state.selected_capacity,
                        position_capacity: state.position_capacity,
                        full_len: state.full_len,
                        raw_len: state.raw_len,
                        pooled_len: state.pooled_len,
                        selected_len: state.selected_len,
                        position: state.position,
                    },
                    input: &scratch.hc_mixed,
                    index_scratch: &scratch.qsa_index,
                    qgate_scratch: &scratch.qsa_qgate,
                    k_scratch: &scratch.qsa_k,
                    v_scratch: &scratch.qsa_v,
                    qsa_output: &scratch.qsa_output,
                    selected_scratch: &scratch.qsa_selected,
                    attention_output: &scratch.attention_output,
                    bf16_scratch: &scratch.gdn_bf16,
                    rows: n,
                    index_heads: dims.indexer_n_heads,
                    index_kv_heads: dims.indexer_kv_heads,
                    index_dim: dims.indexer_head_dim,
                    budget: dims.indexer_budget,
                    compress: dims.indexer_compress_ratio,
                    heads: dims.num_attention_heads,
                    kv_heads: dims.num_key_value_heads,
                    head_dim: dims.head_dim,
                    rotary_dim: config.rotary_dim(),
                    rope: config.rope(),
                    input_width: dims.hidden,
                    trunk_a4: self.trunk_a4.get(layer_index).copied().unwrap_or(0),
                    phase: MixerPhase::Whole,
                }));
            }
        }

        let attn_write = &description.attn_hyper.write;
        steps.push(Step::HyperWrite(HyperWriteOp {
            rotation: &scratch.rotation,
            state_bf16: bf16_state,
            input: self.streams,
            norm_weight: attn_write.norm,
            block_inject: attn_write.block_inject,
            normalized: &scratch.hc_normalized,
            mixed: &scratch.attention_output,
            gates: self.gates,
            output: self.streams,
            rows: n,
            branches: dims.hc_count,
            hidden: config.hidden_size,
        }));

        let mlp_read = &description.mlp_hyper.read;
        steps.push(Step::HyperRead(HyperReadOp {
            rotation: &scratch.rotation,
            state_bf16: bf16_state,
            input: self.streams,
            norm_weight: mlp_read.norm,
            input_mix_down: decode_copy(
                self.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER + 2,
                route_n,
                few_row_limit,
                mlp_read.input_mix_down,
            ),
            input_mix_up: decode_copy(
                self.decode_q8,
                layer_index * DECODE_Q8_PER_LAYER + 3,
                route_n,
                few_row_limit,
                mlp_read.input_mix_up,
            ),
            normalized: &scratch.hc_normalized,
            low: &scratch.hc_low,
            up: &scratch.hc_up,
            mixed: &scratch.hc_mixed,
            rows: n,
            branches: dims.hc_count,
            hidden: config.hidden_size,
            low_rank: dims.hc_lowrank,
        }));
        Ok(())
    }

    /// Seal the layer's MoE call for this program's rows and routing rows
    /// exactly as the incumbent does ([`crate::program::seal_moe_for_layer`]);
    /// `scratch` is the program's layer scratch view.
    pub(crate) fn seal_moe(
        &self,
        ctx: &'a DispatchCtx,
        moe: Qwen4MoeBinding<'a>,
        scratch: &'a Qwen4LayerScratch<'a>,
    ) -> Result<SealedMoeCall<'a>, Qwen4GpuForwardError> {
        crate::program::seal_moe_for_layer(ctx, self.dims, moe, scratch, self.rows, self.route_n)
            .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}")))
    }

    /// The MoE FFN steps of a layer: the routed accumulator clear and the
    /// sealed call (its output is `scratch.ffn_output`).
    pub(crate) fn moe_steps(
        &self,
        steps: &mut Vec<Step<'a>>,
        ctx: &'a DispatchCtx,
        moe: Qwen4MoeBinding<'a>,
        scratch: &'a Qwen4LayerScratch<'a>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let sealed = self.seal_moe(ctx, moe, scratch)?;
        steps.push(Step::Clear(ClearOp {
            tensor: &self.scratch.ffn_output,
            elements: self.rows * self.bundle.config.hidden_size,
        }));
        steps.push(Step::Moe(sealed));
        Ok(())
    }

    /// The dense FFN step (a non-MoE layer): reads `scratch.hc_mixed`, writes
    /// `scratch.ffn_output`. A TP rank's sharded FFN writes its F32 partial
    /// (`Bf16SourcePartial`): the reduction's consumer rounds the sum once.
    pub(crate) fn dense_step(&self, dense: &Qwen4DenseBinding<'a>) -> Step<'a> {
        let scratch = self.scratch;
        Step::GatedMlp(GatedMlpOp {
            rows: self.rows,
            hidden: self.bundle.config.hidden_size,
            intermediate: dense.intermediate,
            x: &scratch.hc_mixed,
            pre_norm: None,
            gate: dense.gate,
            up: dense.up,
            down: dense.down,
            activation: Activation::Silu,
            storage: match self.bundle.weights.tp_shard() {
                Some(_) => MlpStorage::Bf16SourcePartial,
                None => MlpStorage::Bf16Source,
            },
            output: MlpOutput::Write(&scratch.ffn_output),
            ladder: crate::program::QWEN4_DENSE_LADDER,
            scratch: GatedMlpScratch {
                gate: &scratch.moe_gate_batch,
                up: &scratch.moe_up_batch,
                act: &scratch.moe_gate_batch,
                x_rot: &scratch.rotation,
            },
        })
    }

    /// The MLP HC write closing the layer; `mixed` is the FFN output rows
    /// (`scratch.ffn_output`, or the S1 combined-output rows).
    pub(crate) fn mlp_write_step(
        &self,
        description: &Qwen4LayerDescription<'a>,
        mixed: &'a GpuTensor,
    ) -> Step<'a> {
        let scratch = self.scratch;
        let mlp_write = &description.mlp_hyper.write;
        Step::HyperWrite(HyperWriteOp {
            rotation: &scratch.rotation,
            state_bf16: self.bf16_state,
            input: self.streams,
            norm_weight: mlp_write.norm,
            block_inject: mlp_write.block_inject,
            normalized: &scratch.hc_normalized,
            mixed,
            gates: self.gates,
            output: self.streams,
            rows: self.rows,
            branches: self.dims.hc_count,
            hidden: self.bundle.config.hidden_size,
        })
    }
}

/// Parity-harness observer run after each QSA layer's step on the ordinary HIP
/// route: `(gpu, qsa_slot, op)`, with the op's scratch and state holding that
/// layer's inputs, selection and attention output for the forward's rows.
/// Only `reference-parity` builds can install one ([`Qwen4Bundle::set_qsa_tap`]);
/// a forward with a tap never records or replays a retained body.
pub type Qwen4QsaTap =
    Box<dyn FnMut(&mut Gpu, usize, &IndexedAttentionOp<'_>) -> Result<(), String> + Send>;

/// DFlash2 target features of the forwards that run while it is set
/// ([`Qwen4Bundle`]'s capturing spec entries arm and clear it): the completed
/// output of each listed trunk layer, averaged over its hyper-connection
/// streams, into that layer's slot of `rows[row..]` (F32, `layers.len() *
/// hidden` per row). Captured eagerly between the interpreter's steps, so the
/// step program and its fusions are the ones the forward runs without it.
pub(crate) struct Qwen4LayerCapture {
    /// Zero-based trunk layers, ascending; `layer + 1` is never the PLE layer.
    pub(crate) layers: SmallVec<[usize; 8]>,
    /// F32 rows of the whole forward request (a non-owning view).
    pub(crate) rows: GpuTensor,
    /// Request row the running tile starts at (the tile loops set it).
    pub(crate) row: usize,
}

/// The interpreter hook of a [`Qwen4LayerCapture`] forward: right before
/// each target step (the mixer of the layer after a captured one) the
/// streams hold the captured layer's completed output.
struct LayerCaptureHooks<'h> {
    targets: SmallVec<[(*const (), usize); 8]>,
    fired: u32,
    streams: &'h GpuTensor,
    rows: GpuTensor,
    n: usize,
    hc: usize,
    hidden: usize,
    slots: usize,
    bf16: bool,
}

impl LayerCaptureHooks<'_> {
    fn capture(&self, gpu: &mut Gpu, slot: usize) -> Result<(), DispatchError> {
        rdna_compute::tensor_ops::hc_mean_capture_f32(
            gpu,
            self.streams,
            &self.rows,
            self.n,
            self.hc,
            self.hidden,
            self.slots * self.hidden,
            slot * self.hidden,
            self.bf16,
        )
        .map_err(|error| DispatchError::Hip(format!("Qwen4 layer capture: {error}")))
    }

    fn missed(&self) -> bool {
        self.fired != (1u32 << self.targets.len()) - 1
    }
}

impl MoeStepHooks for LayerCaptureHooks<'_> {
    fn before_moe(&mut self, _gpu: &mut Gpu) -> Result<(), DispatchError> {
        Ok(())
    }
    fn after_moe(&mut self, _gpu: &mut Gpu) -> Result<(), DispatchError> {
        Ok(())
    }
    fn before_step(&mut self, gpu: &mut Gpu, step: &Step<'_>) -> Result<(), DispatchError> {
        let at = step as *const Step<'_> as *const ();
        for index in 0..self.targets.len() {
            let (target, slot) = self.targets[index];
            if target == at {
                self.capture(gpu, slot)?;
                self.fired |= 1 << index;
            }
        }
        Ok(())
    }
}

/// Whether a call is a decode-share prompt tile: at least
/// [`QWEN4_DECODE_SHARE_RESIDENT_ROWS`] rows and none of a device-argmax decode
/// forward, a lane step or a mesh segment. The one predicate scratch-bank and
/// BorrowPrefill-runtime selection share.
pub(crate) fn decode_share_prefill_tile(
    rows: usize,
    argmax_forward: bool,
    lane_scope: bool,
    mesh_segment: bool,
) -> bool {
    rows >= QWEN4_DECODE_SHARE_RESIDENT_ROWS && !argmax_forward && !lane_scope && !mesh_segment
}

/// Whether a call inside a whole-prompt entry continues a dirty borrow epoch
/// (the S6b tail rule): the entry is live (`prompt_depth > 0`), it has itself
/// borrowed a prefill bank (`prompt_borrowed`, set by the first prompt tile
/// the entry acquired and cleared when the outermost entry ends) and the donor
/// tail is dirty. A dirty loan left by an earlier, finished prompt (the
/// deferred restore) never continues in a later entry: its first call below
/// the resident rows restores like AR decode, and only a tile of at least the
/// resident rows borrows again.
pub(crate) fn decode_share_continues_epoch(
    prompt_depth: u32,
    prompt_borrowed: bool,
    loan_dirty: bool,
) -> bool {
    prompt_depth > 0 && prompt_borrowed && loan_dirty
}

/// Whether a call borrows a prefill bank and the BorrowPrefill runtimes: a
/// prompt tile ([`decode_share_prefill_tile`]) or a non-empty continuation of
/// the entry's own dirty borrow ([`decode_share_continues_epoch`], `continues`)
/// that is none of a device-argmax decode forward, a lane step or a mesh
/// segment.
pub(crate) fn decode_share_borrows_tile(
    rows: usize,
    argmax_forward: bool,
    lane_scope: bool,
    mesh_segment: bool,
    continues: bool,
) -> bool {
    let continuation = rows > 0 && !argmax_forward && !lane_scope && !mesh_segment && continues;
    continuation || decode_share_prefill_tile(rows, argmax_forward, lane_scope, mesh_segment)
}

/// The per-layer runtimes a tile reads ([`Qwen4GpuForward::tile_moe`]): the
/// BorrowPrefill runtimes of class view `view` while a prompt tile borrows,
/// the Resident ones otherwise.
fn tile_moe_of<'a, T>(
    borrow_active: bool,
    view: usize,
    resident: &'a [T],
    borrowed: &'a [Vec<T>],
) -> &'a [T] {
    match borrowed.get(view) {
        Some(runtimes) if borrow_active => runtimes,
        _ => resident,
    }
}

/// The BorrowPrefill class view a prompt tile reads: the epoch's dirty class,
/// i.e. the largest class any tile of the current borrow epoch (this one
/// included) ran on. `dirty_extent` is the loan's dirty byte prefix after this
/// tile acquired (`None` without a loan: `bind`, no epoch carries over, so the
/// tile's own class). A lease with a single (all-donors) view always reads it.
pub(crate) fn borrow_view_class(
    views: usize,
    tile_class: usize,
    class_spans: &[u64],
    dirty_extent: Option<u64>,
) -> usize {
    if views <= 1 {
        return 0;
    }
    let epoch_class = dirty_extent
        .and_then(|extent| class_spans.iter().position(|&span| span >= extent))
        .unwrap_or(0);
    tile_class.max(epoch_class).min(views - 1)
}

/// Whether the mode runs scratch banks: `bank` over an owned test backing,
/// `fixed` and `on` over the real donor tail.
fn decode_share_banks_requested(mode: crate::decode_share::DecodeShareMode) -> bool {
    use crate::decode_share::DecodeShareMode;
    matches!(
        mode,
        DecodeShareMode::Bank | DecodeShareMode::Fixed | DecodeShareMode::On
    )
}

/// A call's own result merged with the result of its decode-share epilogue:
/// the call's error comes first and keeps its text; a failed epilogue
/// (restore or recovery) is appended, or reported when the call succeeded.
fn merge_share_result(
    result: Result<(), Qwen4GpuForwardError>,
    epilogue: Result<(), Qwen4GpuForwardError>,
) -> Result<(), Qwen4GpuForwardError> {
    match (result, epilogue) {
        (Ok(()), epilogue) => epilogue,
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(extra)) => Err(invalid(format!("{error}; {extra}"))),
    }
}

/// Why a load cannot take the decode-share scratch banks, `None` when the
/// admission basics hold: one band over the whole model, the compact
/// `FnSymGfx1201` plan on gfx1201 (which also rules out the NPU trunk, whose
/// pair executor needs the `Full` plan), a prompt chunk above the resident
/// bank, and no expert parallelism, pipeline/mesh band or slow-link prefill.
fn decode_share_refusal(
    arch: &str,
    mode: Qwen4ForwardScratchMode,
    max_chunk: usize,
    single_scope: bool,
    ep: bool,
    pipeline: bool,
    slowlink: bool,
) -> Option<&'static str> {
    if !single_scope || pipeline {
        Some("a pipeline-parallel or mesh band")
    } else if mode != Qwen4ForwardScratchMode::FnSymGfx1201 {
        Some("the forward scratch is not the compact FnSymGfx1201 plan")
    } else if arch != "gfx1201" {
        Some("the device is not gfx1201")
    } else if max_chunk <= QWEN4_DECODE_SHARE_RESIDENT_ROWS {
        Some("the prompt chunk is not above the resident bank rows")
    } else if ep {
        Some("expert parallelism")
    } else if slowlink {
        Some("S1 slow-link prefill")
    } else {
        None
    }
}

/// The capacity class (index into [`qwen4_decode_share_classes`]) a prompt
/// tile of `rows` rows runs on: the smallest class `>= rows`. `None` for a
/// tile below [`QWEN4_DECODE_SHARE_RESIDENT_ROWS`] (it runs on the resident
/// bank) and for one wider than the largest class (a capacity error).
pub(crate) fn decode_share_class_for_rows(classes: &[usize], rows: usize) -> Option<usize> {
    if rows < QWEN4_DECODE_SHARE_RESIDENT_ROWS {
        return None;
    }
    classes.iter().position(|&class| class >= rows)
}

/// Rows of one prompt tile given the selected bank (`scratch`) and the load's
/// decode-share state: the share's prompt chunk, else the bank's capacity.
fn prompt_chunk_rows_of(
    share: Option<&DecodeShareBanks>,
    scratch: &Qwen4GpuForwardScratch,
) -> usize {
    share.map_or(scratch.max_chunk, |share| share.max_chunk)
}

/// Memory the prefill banks of a decode-share load view.
enum DecodeShareBacking {
    /// `bank` mode: an owned device buffer of the largest class's span (a
    /// correctness mode, charged nowhere).
    Owned(GpuTensor),
    /// `fixed` / `on`: the real donor tail of the expert owner, borrowed under
    /// the loan controller ([`DonorLoan`]). Nothing is allocated or written
    /// at attach.
    Donor { base: usize, bytes: usize },
}

impl DecodeShareBacking {
    fn base(&self) -> usize {
        match self {
            Self::Owned(tensor) => tensor.buf.as_ptr() as usize,
            Self::Donor { base, .. } => *base,
        }
    }

    /// Bytes the banks may span.
    fn bytes(&self) -> usize {
        match self {
            Self::Owned(tensor) => tensor.buf.size(),
            Self::Donor { bytes, .. } => *bytes,
        }
    }
}

/// Counters of a decode-share load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecodeShareStats {
    /// Prompt tiles that ran on a borrowed prefill bank.
    pub tiles_borrowed: u64,
    /// Bank swaps into `scratch` (one per borrowed tile).
    pub bank_swaps: u64,
}

/// The scratch banks of a decode-share load. `Qwen4GpuForward::scratch` is the
/// resident bank (rows [`QWEN4_DECODE_SHARE_RESIDENT_ROWS`]) at every call
/// boundary; a prompt tile of at least that many rows swaps the smallest
/// prefill class that holds it in for the tile and swaps the resident bank
/// back before the call returns.
pub(crate) struct DecodeShareBanks {
    /// Prompt chunk / schedule capacity ([`Qwen4GpuForward::prompt_chunk_rows`]).
    max_chunk: usize,
    /// Capacity of each prefill bank, ascending ([`qwen4_decode_share_classes`]).
    classes: Vec<usize>,
    /// Span bytes of each class: what a tile of that class dirties.
    class_spans: Vec<u64>,
    /// One prefill bank per class, ascending; the active class's slot is empty
    /// while its bank is swapped into `scratch`.
    banks: Vec<Option<Qwen4GpuForwardScratch>>,
    /// The resident bank while a prefill bank is swapped in.
    parked: Option<Qwen4GpuForwardScratch>,
    /// Class index swapped into `scratch`.
    active: Option<usize>,
    /// Persistent `TokenIds` of `max_chunk` rows every bank views.
    token_ids: GpuTensor,
    backing: DecodeShareBacking,
    /// The donor-tail loan (`fixed` / `on`); `None` over an owned test backing.
    loan: Option<DonorLoan>,
    /// Largest span of the classes used since the last restore (a resident call restores
    /// this prefix of the donor tail).
    dirty_bytes: u64,
    stats: DecodeShareStats,
    /// Nesting depth of whole-prompt entries ([`Qwen4GpuForward::share_prompt_begin`]).
    prompt_depth: u32,
    /// The current outermost whole-prompt entry itself acquired a prefill
    /// bank (a tile of at least [`QWEN4_DECODE_SHARE_RESIDENT_ROWS`] rows ran
    /// with `prompt_depth > 0`). Only such an entry continues the dirty borrow
    /// for its own sub-resident tail tiles ([`decode_share_continues_epoch`]);
    /// cleared when the outermost entry ends (whatever its result) and when a
    /// resident call restores the donor tail. A dirty loan left by a finished
    /// prompt must not leak into the next entry's verify, draft or decode calls.
    prompt_borrowed: bool,
}

impl DecodeShareBanks {
    /// Swap the prefill bank of `class` into `scratch`, parking the resident one.
    fn swap_in(&mut self, scratch: &mut Qwen4GpuForwardScratch, class: usize) {
        debug_assert!(self.active.is_none() && self.parked.is_none());
        let bank = self.banks[class]
            .take()
            .expect("a decode-share prefill bank is present when not active");
        self.parked = Some(std::mem::replace(scratch, bank));
        self.active = Some(class);
        self.dirty_bytes = self.dirty_bytes.max(self.class_spans[class]);
        self.stats.tiles_borrowed += 1;
        self.stats.bank_swaps += 1;
    }

    /// Swap the resident bank back into `scratch`; a no-op when none is active.
    fn swap_out(&mut self, scratch: &mut Qwen4GpuForwardScratch) {
        if let (Some(class), Some(resident)) = (self.active.take(), self.parked.take()) {
            self.banks[class] = Some(std::mem::replace(scratch, resident));
        }
    }

    /// Free the persistent token ids, the owned backing and the loan's
    /// resources; the prefill banks own nothing. The resident bank is freed by
    /// its forward. A donor tail is never freed here (the lease owns it).
    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for bank in self.banks.into_iter().flatten().chain(self.parked) {
            if let Some(error) = bank.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some(loan) = self.loan {
            if let Some(error) = loan.release(gpu) {
                first.get_or_insert(error);
            }
        }
        let owned = match self.backing {
            DecodeShareBacking::Owned(backing) => Some(backing),
            DecodeShareBacking::Donor { .. } => None,
        };
        for tensor in owned.into_iter().chain(std::iter::once(self.token_ids)) {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// The restore-completion event. The restore copy runs on the compute stream
/// itself: a dedicated restore stream (even one only ever used for this copy
/// and its events) left every later compute-stream dispatch ~11 us slower for
/// the rest of the process whenever HIP mapped the two streams onto one
/// hardware queue (rocprofv3: median inter-kernel gap 3.1 -> 14.6 us, MTP
/// decode -20%; reproducible every time with `DEBUG_HIP_DYNAMIC_QUEUES=0`).
/// The overlap a second stream would buy is kept anyway: the host does not
/// wait for the copy until the next forward needs the donor tail.
struct LoanHw {
    done: hip_bridge::Event,
    /// Timing events around the epoch's weight-shadow rebuild (only with
    /// borrowed shadows under `HIPFIRE_QWEN4_DECODE_SHARE_LOG=1`): read after
    /// the restore completed.
    rebuild: Option<(hip_bridge::Event, hip_bridge::Event)>,
}

/// The donor-tail loan of a `fixed` / `on` load: the exclusive lease claim, the
/// restore completion event (created at attach, destroyed at free)
/// and the state machine ([`crate::decode_share_loan::LoanController`]).
pub(crate) struct DonorLoan {
    controller: crate::decode_share_loan::LoanController,
    claim: crate::expert_tier_owners::DecodeShareClaim,
    hw: LoanHw,
    /// Device base of the donor tail and its byte extent.
    base: usize,
    bytes: usize,
    /// The donor tail starts with a borrowed weight-shadow region
    /// ([`crate::decode_share_shadows`]) that every epoch's first acquire
    /// rebuilds; the shadow borrow is installed on the `Gpu`.
    shadows: bool,
    /// The byte oracle of the rebuilt shadows already ran for this load.
    shadow_oracle_done: std::cell::Cell<bool>,
    /// Pinned CPU address of the tail's mirror.
    mirror_cpu: usize,
    /// `HIPFIRE_QWEN4_DECODE_SHARE_VERIFY=1`: re-verify the tail after every restore.
    verify: bool,
}

/// [`crate::decode_share_loan::LoanOps`] over one forward's device.
struct GpuLoanOps<'a> {
    gpu: &'a mut Gpu,
    claim: &'a crate::expert_tier_owners::DecodeShareClaim,
    hw: &'a LoanHw,
    shadows: bool,
    shadow_oracle_done: &'a std::cell::Cell<bool>,
    base: usize,
    bytes: usize,
    mirror_cpu: usize,
    verify: bool,
}

impl GpuLoanOps<'_> {
    /// A non-owning view of `[offset, offset + len)` of the donor tail.
    fn tail(&self, offset: u64, len: u64) -> Result<hip_bridge::DeviceBuffer, String> {
        let (offset, len) = (
            usize::try_from(offset).map_err(|_| "donor offset overflows".to_string())?,
            usize::try_from(len).map_err(|_| "donor length overflows".to_string())?,
        );
        if offset.checked_add(len).map_or(true, |end| end > self.bytes) {
            return Err(format!(
                "donor range [{offset}, +{len}) exceeds the {}-byte donor tail",
                self.bytes
            ));
        }
        // SAFETY: the range lies inside the donor tail the claimed lease owns
        // for this forward's life; the buffer is a `Borrowed` alias that no
        // free path ever receives.
        Ok(unsafe {
            hip_bridge::DeviceBuffer::from_raw((self.base + offset) as *mut std::ffi::c_void, len)
        })
    }

    /// The first `len` bytes of the pinned mirror.
    fn mirror(&self, len: u64) -> Result<&[u8], String> {
        let len = usize::try_from(len).map_err(|_| "mirror length overflows".to_string())?;
        if len > self.bytes {
            return Err(format!(
                "mirror range +{len} exceeds the {}-byte tail",
                self.bytes
            ));
        }
        // SAFETY: `mirror_cpu` designates `bytes` bytes of the pinned host owner
        // the claimed lease keeps alive; nothing writes the mirror after
        // materialization.
        Ok(unsafe { std::slice::from_raw_parts(self.mirror_cpu as *const u8, len) })
    }

    fn compute(&self) -> Result<&hip_bridge::Stream, String> {
        self.gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| "decode-share needs the compute stream".to_string())
    }
}

impl crate::decode_share_loan::LoanOps for GpuLoanOps<'_> {
    fn capturing(&mut self) -> Result<bool, String> {
        match self.gpu.active_stream.as_ref() {
            Some(stream) => self
                .gpu
                .hip
                .stream_is_capturing(stream)
                .map_err(|error| error.to_string()),
            None => Ok(false),
        }
    }

    fn memset_zero(&mut self, offset: u64, bytes: u64) -> Result<(), String> {
        let buf = self.tail(offset, bytes)?;
        let compute = self.compute()?;
        self.gpu
            .hip
            .memset_async(&buf, 0, buf.size(), compute)
            .map_err(|error| error.to_string())
    }

    fn enqueue_restore(&mut self, bytes: u64) -> Result<(), String> {
        // The restore overwrites the weight-shadow region: host-side reads of
        // a borrowed shadow are refused from here until the next rebuild.
        // (Stream order keeps every already-enqueued reader ahead of the copy.)
        self.gpu.shadow_borrow_set_ready(false);
        let dst = self.tail(0, bytes)?;
        let src = self.mirror(bytes)?;
        let compute = self.compute()?;
        let hip = &self.gpu.hip;
        let fail = |error: hip_bridge::HipError| error.to_string();
        // Stream order makes the copy land after every scratch consumer of the
        // prompt and before any later compute work; the host wait on `done`
        // is deferred to the next borrowed acquire or resident-bank entry.
        hip.memcpy_htod_async(&dst, src, compute).map_err(fail)?;
        hip.event_record(&self.hw.done, Some(compute)).map_err(fail)
    }

    fn wait_restore(&mut self) -> Result<(), String> {
        self.gpu
            .hip
            .event_synchronize(&self.hw.done)
            .map_err(|error| error.to_string())?;
        self.gpu
            .take_device_wait_error()
            .map_err(|error| error.to_string())
    }

    fn device_sync(&mut self) -> Result<(), String> {
        self.gpu
            .hip
            .device_synchronize()
            .map_err(|error| error.to_string())
    }

    fn blocking_restore(&mut self, bytes: u64) -> Result<(), String> {
        self.gpu.shadow_borrow_set_ready(false);
        let dst = self.tail(0, bytes)?;
        let src = self.mirror(bytes)?;
        self.gpu
            .hip
            .memcpy_htod(&dst, src)
            .map_err(|error| error.to_string())
    }

    fn rebuild_shadows(&mut self) -> Result<(), String> {
        if !self.shadows {
            return Ok(());
        }
        if let Some((start, _)) = self.hw.rebuild.as_ref() {
            let compute = self.compute()?;
            self.gpu
                .hip
                .event_record(start, Some(compute))
                .map_err(|error| error.to_string())?;
        }
        self.gpu
            .rebuild_borrowed_shadows()
            .map_err(|error| error.to_string())?;
        if let Some((_, end)) = self.hw.rebuild.as_ref() {
            let compute = self.compute()?;
            self.gpu
                .hip
                .event_record(end, Some(compute))
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn verify_shadows(&mut self) -> Result<(), String> {
        if !self.shadows || !self.verify || self.shadow_oracle_done.get() {
            return Ok(());
        }
        self.shadow_oracle_done.set(true);
        match self.gpu.verify_borrowed_shadows() {
            Ok(Ok(())) => {
                eprintln!(
                    "  qwen4 decode-share: borrowed weight shadows verified (F16 + INT8) against \
                     an ordinary rebuild into a temporary allocation"
                );
                Ok(())
            }
            Ok(Err(mismatch)) => Err(mismatch),
            Err(error) => Err(error.to_string()),
        }
    }

    fn rebuild_ms(&mut self) -> Option<f64> {
        let (start, end) = self.hw.rebuild.as_ref()?;
        self.gpu
            .hip
            .event_elapsed_ms(start, end)
            .ok()
            .map(f64::from)
    }

    fn verify(&mut self, bytes: u64) -> Result<(), String> {
        if !self.verify {
            return Ok(());
        }
        let prefix = usize::try_from(bytes).map_err(|_| "verify prefix overflows".to_string())?;
        self.claim
            .lease()
            .verify_donor_tail(&mut *self.gpu, prefix)
            .map(|_| ())
    }
}

impl DonorLoan {
    /// Run `f` over the controller and the device operations.
    fn with_ops<R>(
        &mut self,
        gpu: &mut Gpu,
        f: impl FnOnce(&mut crate::decode_share_loan::LoanController, &mut GpuLoanOps<'_>) -> R,
    ) -> R {
        let mut ops = GpuLoanOps {
            gpu,
            claim: &self.claim,
            hw: &self.hw,
            base: self.base,
            bytes: self.bytes,
            mirror_cpu: self.mirror_cpu,
            verify: self.verify,
            shadows: self.shadows,
            shadow_oracle_done: &self.shadow_oracle_done,
        };
        f(&mut self.controller, &mut ops)
    }

    /// Destroy the restore event, then release the claim. Call only after a
    /// successful teardown restore (the event has completed).
    fn release(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        if let Some((start, end)) = self.hw.rebuild {
            for event in [start, end] {
                if let Err(error) = gpu.hip.event_destroy(event) {
                    first.get_or_insert(error);
                }
            }
        }
        if self.shadows {
            // Never frees the region: the donor tail belongs to the lease.
            gpu.clear_shadow_borrow();
        }
        if let Err(error) = gpu.hip.event_destroy(self.hw.done) {
            first.get_or_insert(error);
        }
        first
    }
}

/// Reusable production forward owner.  Construct it once per assembled bundle
/// and use `forward_token`/`forward_chunk` repeatedly; both methods share the
/// exact same chunk implementation.
pub struct Qwen4GpuForward {
    pub scratch: Qwen4GpuForwardScratch,
    pub(crate) host_token_bytes: Vec<u8>,
    /// Rows of `scratch.token_ids` the last forward uploaded whole from
    /// `host_token_bytes` (its mirror): 0 for a HIP single-row body, a device
    /// argmax id or a captured upload.
    pub(crate) uploaded_id_rows: usize,
    /// Host staging for the PLE rows and the fence over its asynchronous
    /// upload (`HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD`).
    pub(crate) host_ple: PleHostStage,
    /// Tokens of the prefill chunk that follows the next forward. A
    /// successful forward enqueues their PLE rows (hashed from the history it
    /// committed) as `ple_ahead` once its own lease is released, so the next
    /// chunk's reads and row materialization overlap this chunk's GPU work.
    /// Consumed (cleared) by every forward.
    ple_lookahead: Vec<u32>,
    /// Retained next-chunk PLE ticket. The next forward adopts it when its
    /// row ids and epoch match exactly, else drops it and fetches afresh.
    pub(crate) ple_ahead: Option<RowTicket>,
    /// Device-argmax token readback (`forward_token_or_argmax`), once used.
    token_readback: Option<TokenReadback>,
    /// `HIPFIRE_QWEN4_TOKEN_READBACK_EARLY` (default on), read once at
    /// construction: whether `TokenReadback` queues the id copy at the
    /// argmax instead of at the PLE read.
    token_readback_early: bool,
    /// MoE runtime of each absolute trunk layer (`None`: a layer without a
    /// MoE FFN or one this band does not own); length `num_hidden_layers`.
    pub(crate) moe: Vec<Option<Qwen4MoeLayerRuntime>>,
    cpu_deferred_experts: bool,
    /// Trunk layers armed for the D2 tier split (a distinct numerical mode:
    /// never counted in `cpu_deferred_experts`).
    cpu_split_layers: usize,
    /// The D2 split was requested by an explicit `memory.offload_exec=cpu`
    /// (width caps and eager rules for the whole forward), not by the
    /// per-load measured default (which affects only the calls that execute it).
    cpu_split_explicit: bool,
    expert_stage: Option<Qwen4ExpertStage>,
    /// Expert-parallel state (`Some` iff the bundle's weights carry an EP
    /// placement): the zero dummy expert, the full per-layer tables for chunk
    /// forwards, the row-gather buffers and the installed gather. Under EP
    /// `moe` holds the per-layer decode tables, there is no G2 expert stage,
    /// and no forward records or replays a retained body.
    pub(crate) ep: Option<Qwen4ForwardEp>,
    /// `DECODE_Q8_PER_LAYER` per layer [HC attn down, attn up, mlp down, mlp
    /// up, shared gate, up, down]: BF16 matrices requantized to Q8_0 for
    /// single-token forwards (gfx1151). Batch-1 decode streams every weight
    /// once per token, and these are ~1.7 GB of it; Q8_0 halves that at a
    /// decode KLD inside the noise band (HC 0.0743 -> 0.0754, shared +0.0007
    /// over 32 chunks). Prefill keeps the BF16 source for its WMMA routes and
    /// the exact multi-row arms short prompts take.
    pub(crate) decode_q8: Vec<GpuTensor>,
    /// `HIPFIRE_QWEN4_ROUTE_TRACE`: routed expert ids per layer of every
    /// single-token HIP forward, for expert-cache sizing. `None` when unset.
    route_trace: Option<RouteTrace>,
    pub(crate) qsa_tap: Option<Qwen4QsaTap>,
    /// DFlash2 target-feature capture of the forwards that run while it is
    /// set; `None` everywhere else (see [`Qwen4LayerCapture`]).
    pub(crate) layer_capture: Option<Qwen4LayerCapture>,
    /// `HIPFIRE_QWEN4_CUT_LAYERS`, read once at construction: when set, the
    /// program runs as separate interpreter runs split at these layers (eager
    /// only; no retained replay).
    cut_layers: Option<CutLayers>,
    /// `HIPFIRE_QWEN4_TRUNK_IU4`: per layer, the GDN Z|beta|alpha fold the
    /// IU4 trunk's V2B SET reads (see [`prepare_trunk_iu4`]); empty when the
    /// route is off or the trunk did not verify.
    pub(crate) trunk_zba: Vec<Option<GpuTensor>>,
    /// `HIPFIRE_QWEN4_TRUNK_IU4`: per layer, the [`TrunkFamily`] bits the
    /// mask selected for A4 (empty when the route is off or did not arm).
    pub(crate) trunk_a4: Vec<u16>,
    /// Set only while a lane's prefill tile runs the singleton chunk program
    /// on a lane state ([`Qwen4Bundle::lane_prefill_tile`]): the forward is
    /// then never a retained-body candidate, so it neither records nor
    /// replays the singleton's tape (whose pointers name the resident state).
    pub(crate) lane_scope: bool,
    /// Set only by the bundle's prefill entries (`forward_chunk*`,
    /// `spec_prefill_rows*`, `lane_prefill_tile`) around their forward call:
    /// the batch-invariant prefill profile ([`Gpu::prefill_invariant_scope`])
    /// runs always on gfx942 and with `HIPFIRE_PREFILL_PM_INVARIANT` on gfx1151. Never
    /// set for decode, MTP verify or the MTP head.
    pub(crate) prefill_scope: bool,
    /// Lab receipt of the reference actually bound by the most recent
    /// multi-row verify attempt, retained after the temporary GPU scope exits.
    #[cfg(feature = "lab")]
    pub(crate) last_verify_reference: Option<rdna_compute::prefill_invariant::Qwen4VerifyReference>,
    /// The band this forward runs: the weights' and the state's scope (the
    /// whole model is [`Qwen4BandScope::single`]).
    scope: Qwen4BandScope,
    /// `Some(route_rows)` only inside [`Self::forward_mesh_chunk`]: the epoch's
    /// canonical routing rows. Mesh forwards are enqueue-only: no retained
    /// body, no blocking token or PLE upload, no context growth.
    mesh_route_rows: Option<usize>,
    /// `Some` only inside [`Self::forward_mesh_chunk`]: the segment it runs.
    mesh_segment: Option<MeshSegment>,
    /// The first op of the next segment while a mesh band program is part
    /// way through; `None` between programs.
    mesh_next_op: Option<u32>,
    /// Fenced async upload of a mesh input band's token ids.
    mesh_tokens: PleHostStage,
    /// Pinned, fenced ring for the lane forward's token ids (the active-stream
    /// asynchronous upload; capture keeps the blocking `host_token_bytes` copy).
    pub(crate) lane_tokens: crate::lane_token_ring::LaneTokenRing,
    /// Op ids of this band's trunk program (see [`mesh_trunk_op_range`]).
    mesh_trunk_ops: std::ops::Range<u32>,
    /// S1 slow-link layer-major prefill state: the admitted profile and, once
    /// the first S1 prompt ran, its page ring and prompt storage
    /// ([`crate::slowlink_prefill`]). Empty on every other load.
    pub(crate) slowlink: crate::slowlink_prefill::SlowlinkSlot,
    #[cfg(target_os = "linux")]
    npu_trunk: Option<npu_forward::NpuForward>,
    /// `Some` on a decode-share load: the prefill banks and bank-swap state.
    share: Option<DecodeShareBanks>,
    /// BorrowPrefill twins of `moe` per prefill capacity class (each in the
    /// same layer order; empty unless the tier lease carries a decode-share
    /// binding and the mode uses it). Class `c` reads the donors its scratch
    /// can overwrite at their pinned mirror and every other donor in VRAM.
    moe_borrow: Vec<Vec<Option<Qwen4MoeLayerRuntime>>>,
    /// Row capacity of each class ([`qwen4_decode_share_classes`]) the
    /// BorrowPrefill views were built for (empty without them).
    borrow_class_rows: Vec<usize>,
    /// Set only by the decode-share path ([`Self::forward_chunk_inner`]) for
    /// the duration of one prompt tile: [`Self::tile_moe`] then reads
    /// `moe_borrow[borrow_view]`.
    borrow_runtimes_active: bool,
    /// The class view of the running prompt tile.
    borrow_view: usize,
    /// `Some` on an `HIPFIRE_EP_RESERVE_DECOUPLE=1` load: the 4096-row prompt plan,
    /// the resident / borrowed scratch banks, the donor loan and the epoch proof
    /// ([`ep_reserve_forward`]). `scratch` is the resident bank at every call
    /// boundary; the borrowed runtime view is `moe_borrow[0]`.
    ep_reserve: Option<ep_reserve_forward::EpReserveForward>,
    /// `Some` on an ORDINARY EP load under `HIPFIRE_EP_RESERVE_RECEIPT=1` (no
    /// reserve): the host-only prompt-record state of the gate's off arm.
    ep_ordinary_receipt: Option<ep_reserve_forward::EpOrdinaryReceipt>,
    /// Set only across [`Self::share_prompt_begin_mtp_fill`]'s begin: the
    /// entry it opens is the native-MTP fill (receipt `phase`).
    receipt_mtp_fill: bool,
}

/// An event after a forward's device argmax and a stream independent of the
/// legacy null stream to read the token id on.
struct TokenReadback {
    stream: hip_bridge::Stream,
    event: hip_bridge::Event,
    /// `HIPFIRE_QWEN4_TOKEN_READBACK_EARLY`: pinned 4-byte landing for the
    /// id copy that `enqueue` queues on `stream` right behind the argmax
    /// event. `None` keeps the legacy sequence (host event wait, then a D2H
    /// into a stack array, then a stream wait) at `read_token`.
    pinned: Option<hip_bridge::PinnedTransferBuffer>,
}

// SAFETY: `TokenReadback` exclusively owns its stream, event and pinned id
// buffer. `Stream` and `Event` are already `Send`; the pinned buffer's raw
// stream handle is the same kind of opaque HIP handle, its `hipHostMalloc`
// allocation has a single owner (no aliasing across threads) and its HIP
// function pointers/library are thread-agnostic. Every use goes through
// `&mut self`, so there is no concurrent access, and teardown drains the
// stream first (`Qwen4GpuForward::free_gpu`, `PinnedTransferBuffer::drop`).
unsafe impl Send for TokenReadback {}

impl TokenReadback {
    /// Stream, event and (when `early`) the pinned id buffer. A failure
    /// destroys whatever this call already created.
    fn create(hip: &hip_bridge::HipRuntime, early: bool) -> Result<Self, hip_bridge::HipError> {
        let stream = hip.stream_create_non_blocking()?;
        let event = match hip.event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING) {
            Ok(event) => event,
            Err(error) => {
                let _ = hip.stream_destroy(stream);
                return Err(error);
            }
        };
        let pinned = if early {
            match hip.transfer_buffer(std::mem::size_of::<i32>()) {
                Ok(buffer) => Some(buffer),
                Err(error) => {
                    let _ = hip.event_destroy(event);
                    let _ = hip.stream_destroy(stream);
                    return Err(error);
                }
            }
        } else {
            None
        };
        Ok(Self {
            stream,
            event,
            pinned,
        })
    }

    /// Record the argmax-done event on the compute stream. Early mode then
    /// makes the readback stream wait on that event and queues the 4-byte id
    /// copy into the pinned buffer, so `read_token` only drains this stream.
    /// The copy reads `token_ids` after the event, i.e. the argmax this call
    /// follows; every later writer of `token_ids` first runs
    /// [`Self::complete`].
    fn enqueue(
        &mut self,
        hip: &hip_bridge::HipRuntime,
        token_ids: &hip_bridge::DeviceBuffer,
        active: Option<&hip_bridge::Stream>,
    ) -> Result<(), hip_bridge::HipError> {
        hip.event_record(&self.event, active)?;
        if let Some(pinned) = self.pinned.as_mut() {
            hip.stream_wait_event(&self.stream, &self.event)?;
            pinned.download_async(
                hip,
                token_ids,
                0,
                std::mem::size_of::<i32>(),
                Some(&self.stream),
            )?;
        }
        Ok(())
    }

    /// Drain a queued id copy (no-op when none is pending), so `token_ids`
    /// may be rewritten and the pinned bytes read.
    fn complete(&mut self, hip: &hip_bridge::HipRuntime) -> Result<(), hip_bridge::HipError> {
        match self.pinned.as_mut() {
            Some(pinned) => pinned.complete(hip),
            None => Ok(()),
        }
    }

    /// The id of the argmax `enqueue` followed. Early mode drains the queued
    /// copy and reads the pinned bytes; otherwise the legacy event wait, D2H
    /// and stream wait.
    fn read_token(
        &mut self,
        hip: &hip_bridge::HipRuntime,
        token_ids: &hip_bridge::DeviceBuffer,
    ) -> Result<u32, hip_bridge::HipError> {
        let mut bytes = [0u8; 4];
        if let Some(pinned) = self.pinned.as_mut() {
            pinned.complete(hip)?;
            bytes.copy_from_slice(&pinned.as_bytes_mut()[..4]);
        } else {
            hip.event_synchronize(&self.event)?;
            hip.memcpy_dtoh_async(&mut bytes, token_ids, Some(&self.stream))?;
            hip.stream_synchronize(&self.stream)?;
        }
        Ok(i32::from_ne_bytes(bytes) as u32)
    }
}

/// `HIPFIRE_QWEN4_CUT_LAYERS`: developer knob naming the absolute trunk layer
/// indices at which a new band starts (the cut-safe Single schedule).
pub(crate) const CUT_LAYERS_ENV: &str = "HIPFIRE_QWEN4_CUT_LAYERS";

/// At most seven cuts (eight bands).
pub(crate) const MAX_CUT_LAYERS: usize = 7;

/// Validated, strictly increasing cut layers; fixed storage, no heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CutLayers {
    layers: [usize; MAX_CUT_LAYERS],
    len: usize,
}

impl CutLayers {
    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.layers[..self.len]
    }

    fn contains(&self, layer: usize) -> bool {
        self.as_slice().contains(&layer)
    }
}

/// Parses `HIPFIRE_QWEN4_CUT_LAYERS`: 1..=7 comma-separated, strictly
/// increasing layer indices, each `>= 1`, `< num_hidden_layers`, and past the
/// PLE layer when the profile has one (the PLE layer stays in the first band).
pub(crate) fn parse_qwen4_cut_layers(
    raw: &str,
    num_hidden_layers: usize,
    ple_layer_index: Option<usize>,
) -> Result<CutLayers, String> {
    let mut cuts = CutLayers {
        layers: [0; MAX_CUT_LAYERS],
        len: 0,
    };
    for entry in raw.split(',') {
        let entry = entry.trim();
        let layer: usize = entry
            .parse()
            .map_err(|_| format!("`{entry}` is not a layer index"))?;
        if cuts.len == MAX_CUT_LAYERS {
            return Err(format!("more than {MAX_CUT_LAYERS} cut layers"));
        }
        if layer == 0 {
            return Err("cut layer 0 is not a cut (layers start at 1)".to_string());
        }
        if layer >= num_hidden_layers {
            return Err(format!(
                "cut layer {layer} is outside the {num_hidden_layers}-layer trunk"
            ));
        }
        if let Some(ple) = ple_layer_index {
            if layer <= ple {
                return Err(format!(
                    "cut layer {layer} must be after the PLE layer {ple}"
                ));
            }
        }
        if cuts.len > 0 && layer <= cuts.layers[cuts.len - 1] {
            return Err(format!(
                "cut layer {layer} is not strictly after {}",
                cuts.layers[cuts.len - 1]
            ));
        }
        cuts.layers[cuts.len] = layer;
        cuts.len += 1;
    }
    Ok(cuts)
}

/// Segment boundaries of the cut-safe schedule: the PLE split (when present,
/// always before every cut) then the recorded cut step indices.
fn cut_boundaries(
    ple_split: Option<usize>,
    cut_steps: &[usize],
) -> ([usize; MAX_CUT_LAYERS + 1], usize) {
    let mut bounds = [0usize; MAX_CUT_LAYERS + 1];
    let mut count = 0usize;
    if let Some(split) = ple_split {
        debug_assert!(cut_steps.first().map_or(true, |&cut| split < cut));
        bounds[0] = split;
        count = 1;
    }
    for &cut in cut_steps {
        bounds[count] = cut;
        count += 1;
    }
    (bounds, count)
}

/// Op ids of the trunk (no head) the band runs, in issue order, from the pure
/// step graph: the input `Embed` op leads the input band and the head ops
/// ([`Qwen4GpuForward::mesh_op_range`]) trail the output band's range.
fn mesh_trunk_op_range(
    config: &Qwen4Config,
    scope: &Qwen4BandScope,
) -> Result<std::ops::Range<u32>, Qwen4GpuForwardError> {
    let graph =
        crate::program::qwen4_step_graph_with_output(config, MeshRoute::Decode, false, false)
            .map_err(invalid)?;
    let ids = crate::program::qwen4_rank_op_ids(&graph, scope);
    let (Some(first), Some(last)) = (ids.first(), ids.last()) else {
        return Err(invalid("the band owns no step-graph op"));
    };
    if (last.0 - first.0) as usize + 1 != ids.len() {
        return Err(invalid("the band's step-graph ops are not contiguous"));
    }
    Ok(first.0..last.0 + 1)
}

/// Refuse, before any allocation, a band forward this build cannot run:
/// the state must be scoped exactly like the weights, and a non-single band
/// has no expert-parallel, tiered-expert or host-mapped route.
fn check_band_admission(
    bundle: &Qwen4Bundle,
    scope: &Qwen4BandScope,
    single_scope: bool,
) -> Result<(), Qwen4GpuForwardError> {
    scope.validate(&bundle.config).map_err(invalid)?;
    let map = bundle.state.scope_map();
    if map.band() != scope.band
        || map.owns_input() != scope.input
        || map.owns_output() != scope.output
    {
        return Err(invalid(format!(
            "band {:?} (input {}, output {}) disagrees with state {:?}",
            scope.band, scope.input, scope.output, map,
        )));
    }
    if !single_scope
        && (bundle.weights.expert_tiers().is_some() || bundle.weights.ep_placement().is_some())
    {
        return Err(invalid(
            "a pipeline-parallel band refuses expert tiers and expert parallelism",
        ));
    }
    Ok(())
}

impl Qwen4GpuForward {
    /// Live D2 executor capacity for the bundle's common cap: the minimum allocated
    /// capacity across armed split layers, `None` when no layer owns an executor.
    /// `None` means D2 contributes nothing and must not clip a D1 64.
    pub(crate) fn d2_storage_capacity_rows(&self) -> Option<usize> {
        let mut min: Option<usize> = None;
        for layer in self.moe.iter().flatten() {
            let Some(executor) = layer.cpu_split.as_ref() else {
                continue;
            };
            min = Some(min.map_or(executor.capacity_rows(), |m: usize| {
                m.min(executor.capacity_rows())
            }));
        }
        min
    }
}
/// Environment knob naming the routing-trace output file (appended).
pub const ROUTE_TRACE_ENV: &str = "HIPFIRE_QWEN4_ROUTE_TRACE";

/// Routing trace sink: one text line per single-token forward holding
/// `position` then `num_hidden_layers * num_experts_per_tok` expert ids in
/// layer order. Each layer's top-K ids are copied device-to-device right after
/// its MoE step, so tracing splits the step list at MoE barriers only.
struct RouteTrace {
    device: GpuTensor,
    host: Vec<u8>,
    out: std::io::BufWriter<std::fs::File>,
}

/// Dense IU4 trunk (`HIPFIRE_QWEN4_TRUNK_IU4=<mask>`, see
/// [`rdna_compute::Qwen4TrunkMask`]) over a trunk that may mix symmetric
/// MQ4G256V2 (QT44) and other tiers (MQ6 ...) per tensor: check on the device
/// that every MQ4G256V2 trunk projection (GDN qkv/z/a/b/out, QSA
/// indexer/q/k/v/o) is a symmetric QT44 matrix, then fold each GDN layer's Z,
/// beta (`in_proj_b`) and alpha (`in_proj_a`) rows plus zero rows into one
/// `z.m + 256`-row matrix for the dense V2B Z|beta|alpha SET, for the layers
/// where all three are MQ4 and in the mask (the rows are copied verbatim; the
/// original tensors stay for decode). Arms `gpu.qwen4_trunk_iu4` iff at least
/// one trunk tensor is MQ4G256V2 and in the mask (A4 projections > 0) and
/// every MQ4 one verified. Returns the folds
/// and the per-layer A4 mask bits (empty when not armed). The mask is the
/// `HIPFIRE_QWEN4_TRUNK_IU4` override when that variable is set (`0` = off),
/// else the artifact's embedded `qwen4_trunk_iu4_default` default on its exact
/// architecture ([`rdna_compute::qwen4_trunk_iu4_route`]); an unmarked
/// artifact is off unless the variable selects a mask. A bad mask value
/// fails the load even when the route cannot arm; nothing else runs unless
/// the route was requested.
fn prepare_trunk_iu4(
    gpu: &mut Gpu,
    bundle: &Qwen4Bundle,
) -> Result<(Vec<Option<GpuTensor>>, Vec<u16>), Qwen4GpuForwardError> {
    gpu.qwen4_trunk_iu4 = false;
    let route = rdna_compute::qwen4_trunk_iu4_route(&gpu.arch, bundle.config.trunk_iu4_default)
        .map_err(invalid)?;
    let mask = &route.mask;
    mask.check_layers(bundle.config.num_hidden_layers)
        .map_err(|error| invalid(format!("HIPFIRE_QWEN4_TRUNK_IU4: {error}")))?;
    let mut folds = Vec::new();
    if !gpu.qwen4_trunk_iu4_requested(mask) {
        // A marked artifact on its exact arch whose default the env switched
        // off (`0` / empty): say so, so the kill is visible at load. Unmarked
        // artifacts and other arches return silently, as before.
        if let Some(default) = bundle.config.trunk_iu4_default {
            if route.source == rdna_compute::TrunkIu4MaskSource::Env
                && mask.is_empty()
                && gpu.arch == default.arch()
            {
                eprintln!(
                    "  qwen4 IU4 trunk: mask=off (env override), artifact default {} disabled; QT44 trunk weights stay resident, stock route",
                    default.marker()
                );
            }
        }
        return Ok((folds, Vec::new()));
    }
    let weights = &bundle.weights;
    let (mut mq4, mut verified, mut a4, mut other) = (0usize, 0usize, 0usize, 0usize);
    for layer in weights.layer_refs.iter() {
        let bits = mask.bits(layer.layer);
        let matrices = match (&layer.gdn, &layer.attention) {
            (Some(gdn), _) => {
                let g = gdn_desc(weights, gdn)?;
                [
                    (TrunkFamily::GdnQkv, g.qkv),
                    (TrunkFamily::GdnZ, g.z),
                    (TrunkFamily::GdnA, g.in_proj_a),
                    (TrunkFamily::GdnB, g.in_proj_b),
                    (TrunkFamily::GdnOut, g.output),
                ]
            }
            (None, Some(qsa)) => {
                let q = qsa_desc(weights, qsa)?;
                [
                    (TrunkFamily::QsaIdx, q.indexer_qk),
                    (TrunkFamily::QsaQ, q.q),
                    (TrunkFamily::QsaK, q.k),
                    (TrunkFamily::QsaV, q.v),
                    (TrunkFamily::QsaO, q.output),
                ]
            }
            (None, None) => return Err(invalid("trunk layer without attention weights")),
        };
        for (family, w) in matrices {
            if w.dtype != DType::MQ4G256V2 {
                other += 1;
                continue;
            }
            mq4 += 1;
            if gpu.qwen4_trunk_sym_check(w.buf, w.m, w.k)? {
                verified += 1;
            }
            if bits & family.bit() != 0 {
                a4 += 1;
            }
        }
    }
    eprintln!(
        "  qwen4 IU4 trunk: mask={} ({}) MQ4 {verified}/{mq4} verified symmetric, A4 {a4} projections, MQ6/other {other}",
        mask.canonical(),
        match route.source {
            rdna_compute::TrunkIu4MaskSource::Env => "env override".to_owned(),
            rdna_compute::TrunkIu4MaskSource::ArtifactDefault(default) =>
                format!("artifact default {}", default.marker()),
            rdna_compute::TrunkIu4MaskSource::Off => "off".to_owned(),
        }
    );
    if mq4 == 0 || verified != mq4 || a4 == 0 {
        return Ok((folds, Vec::new()));
    }
    let result = (|| {
        for index in 0..bundle.config.num_hidden_layers {
            let Some(gdn) = weights.layer(index).and_then(|layer| layer.gdn.as_ref()) else {
                folds.push(None);
                continue;
            };
            let g = gdn_desc(weights, gdn)?;
            let k = g.z.k;
            let zba = TrunkFamily::GdnZ.bit() | TrunkFamily::GdnA.bit() | TrunkFamily::GdnB.bit();
            if mask.bits(index) & zba != zba
                || [g.z.dtype, g.in_proj_a.dtype, g.in_proj_b.dtype]
                    .iter()
                    .any(|d| *d != DType::MQ4G256V2)
                || g.in_proj_a.k != k
                || g.in_proj_b.k != k
                || g.in_proj_a.m != 48
                || g.in_proj_b.m != 48
            {
                folds.push(None);
                continue;
            }
            let row_bytes = k / 256 * rdna_compute::MQ4V2_GROUP_BYTES;
            let mut rows = vec![0u8; (g.z.m + 256) * row_bytes];
            for (w, start) in [(&g.z, 0), (&g.in_proj_b, g.z.m), (&g.in_proj_a, g.z.m + 48)] {
                let span = start * row_bytes..(start + w.m) * row_bytes;
                gpu.hip.memcpy_dtoh(&mut rows[span], &w.buf.buf)?;
            }
            folds.push(Some(gpu.upload_raw(&rows, &[rows.len()])?));
        }
        Ok::<(), Qwen4GpuForwardError>(())
    })();
    if let Err(error) = result {
        for tensor in folds.into_iter().flatten() {
            let _ = gpu.free_tensor(tensor);
        }
        return Err(error);
    }
    // Every MQ4V2 matrix the Qwen4 forward hands the dense IU4 GEMMs is a
    // trunk projection verified above (routed experts take their own verified
    // route), so the symmetric-grid contract those SET kernels read
    // (`mq4v2_symmetric`: V2B / X5 / symfold) holds even though Qwen4
    // artifacts record expert symmetry under their own metadata key.
    gpu.mq4v2_symmetric = true;
    gpu.qwen4_trunk_iu4 = true;
    let layer_bits = (0..bundle.config.num_hidden_layers)
        .map(|layer| mask.bits(layer))
        .collect();
    Ok((folds, layer_bits))
}

/// Populate absolute runtime slots from the assembled, rank-local layer refs.
/// Keep each successful runtime in its slot immediately so the caller's error
/// cleanup also owns runtimes built before a later layer fails. The pure
/// selection seam lets CPU tests exercise PP ownership without GPU tensors.
fn populate_moe_runtimes<T, E>(
    runtimes: &mut [Option<T>],
    layers: &[Qwen4LayerWeights],
    mut build: impl FnMut(usize, &MoeWeights) -> Result<T, E>,
) -> Result<(), E> {
    for layer in layers {
        if let Some(weights) = layer.ffn.moe() {
            runtimes[layer.layer] = Some(build(layer.layer, weights)?);
        }
    }
    Ok(())
}

impl Qwen4GpuForward {
    /// `(verified, total)` routed trunk MoE layers whose final bound pointer tables passed the
    /// load-time header check ([`Qwen4MoeLayerRuntime::verify_symmetric`], host-mapped S1 layers
    /// included): the receipt's `moe_iu4.final_headers_verified`. `(0, total)` when the route
    /// was never requested.
    pub(crate) fn sym_iu4_coverage(&self) -> (usize, usize) {
        let total = self.moe.iter().flatten().count();
        let verified = self
            .moe
            .iter()
            .flatten()
            .filter(|layer| layer.symmetric)
            .count();
        (verified, total)
    }
    /// Whether host arithmetic runs on every call of this load (the D1
    /// whole-layer splice, or an explicit-mode D2 split;
    /// [`host_arith_every_call`]). Load-wide admission (row cap, eager,
    /// replay/capture refusal, speculative width) keys on this; the
    /// automatic D2 default does not, it is per call
    /// ([`Self::call_executes_host_arith`]). The D1-only per-layer flag stays
    /// `Qwen4MoeLayerRuntime::cpu_deferred_experts`.
    pub(crate) fn has_cpu_deferred_experts(&self) -> bool {
        host_arith_every_call(
            self.cpu_deferred_experts,
            self.cpu_split_layers > 0,
            self.cpu_split_explicit,
        )
    }
    /// Whether a call of `rows` rows over `route_rows` routing rows in the
    /// given phase executes host arithmetic ([`host_arith_call`]). `gpu`
    /// supplies this call's frozen row-exact width; capacity is the minimum
    /// allocated [`capacity_rows`](hipfire_dispatch::cpu_split::CpuSplitExecutor::capacity_rows)
    /// across armed layers, so a wide window runs only when every layer that
    /// owns it has storage for it.
    /// `lab_armed` is the caller-observed existing-oracle-override presence
    /// for this call (env or runtime override on an exact oracle arch with an
    /// eager verify scope) — never derived from `few_row_limit` here (that is
    /// a GPU projection-route limit, not CPU storage). Production threads it
    /// from the forward's eager-verify scope; tests pass it explicitly.
    /// `dim/mi/top_k` are this call's expert geometry (the D1 capacity
    /// source reads actual prewarmed rows for them); `device_id` is the
    /// sealing device. Tests pass explicit values.
    pub(crate) fn call_executes_host_arith(
        &self,
        gpu: &Gpu,
        prefill: bool,
        rows: usize,
        route_rows: usize,
        lab_armed: bool,
        dim: usize,
        mi: usize,
        top_k: usize,
    ) -> bool {
        if host_arith_call(
            self.has_cpu_deferred_experts(),
            self.cpu_split_layers > 0,
            prefill,
            rows,
            route_rows,
            lab_armed,
            qwen4_row_exact_verify_rows(gpu),
            self.min_split_capacity_rows(),
        ) {
            return true;
        }
        // D1 wide per-call path (no split executor): an armed D1 layer runs
        // host arithmetic on a whole logical 1..64-row eager-verify window
        // when the lab is armed and host scratch is actually prewarmed.
        // Default (unarmed, tail, or no prewarm) stays false.
        if prefill || !lab_armed || !self.cpu_deferred_experts {
            return false;
        }
        let scope = hipfire_dispatch::families::moe::CpuWideVerifyScope::from_lab_presence(
            lab_armed,
            qwen4_row_exact_verify_rows(gpu),
        );
        hipfire_dispatch::families::moe::cpu_d1_wide_window_qualified(
            true,
            rows,
            route_rows,
            scope,
            hipfire_dispatch::families::moe::cpu_d1_prewarmed_scratch_rows(
                dim,
                mi,
                top_k,
                gpu.device_id,
            ),
        )
    }

    /// Existing-oracle-override presence for this GPU: the canonical
    /// [`rdna_compute::prefill_invariant::qwen4_wide_verify_override_present`]
    /// signal (runtime override — including the `Some(8)` baseline — or the
    /// parsed developer var, on an exact oracle arch). Pure presence signal
    /// for the CPU experimental capacity gate — never a capacity value, never
    /// `few_row_limit` (that is a GPU projection-route limit, not CPU
    /// storage). Tests pass `lab_armed` explicitly instead of reading it here.
    pub(crate) fn cpu_lab_override_present(gpu: &Gpu) -> bool {
        rdna_compute::prefill_invariant::qwen4_wide_verify_override_present(&gpu.arch)
    }

    /// The S1 admission verdict of every trunk MoE layer, `(label, verdict)`
    /// in layer order: the head's runtime is never in this list.
    pub(crate) fn s1_layer_verdicts(&self) -> Vec<(usize, Result<(), &'static str>)> {
        self.moe
            .iter()
            .flatten()
            .map(|layer| (layer.label, layer.s1_layer_check()))
            .collect()
    }

    /// Minimum allocated executor capacity across armed split layers (`0`
    /// when none is armed). Internal routing reader for the per-call D2 gate
    /// (unarmed callers fail closed on 0). Bundle's common cap uses
    /// [`Self::d2_storage_capacity_rows`] instead (`None` = unowned, must not
    /// clip a D1 64). Same instances, distinct contracts — not duplicates.
    pub(crate) fn min_split_capacity_rows(&self) -> usize {
        self.moe
            .iter()
            .flatten()
            .filter_map(|layer| layer.cpu_split.as_ref())
            .map(|executor| executor.capacity_rows())
            .min()
            .unwrap_or(0)
    }
    /// Verify rows the armed CPU numerical modes run row-exact, read from each
    /// mode's own ownership and exported window, **independent of**
    /// [`Self::has_cpu_deferred_experts`]: the D1 whole-layer splice
    /// (`cpu_deferred_experts`, [`cpu_splice::NARROW_VERIFY_MAX_ROWS`]) and any
    /// armed D2 tier split (`cpu_split_layers > 0`,
    /// [`hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS`]), including the
    /// measured per-load D2 default that [`host_arith_every_call`] excludes
    /// from the load-wide predicate: that default withholds its CPU executor
    /// from a window wider than the narrow cap ([`d2_executor_admitted`]), so a
    /// wider verify would run GPU arithmetic against CPU-arithmetic decode
    /// rows, and the `auto` cap must not draft past the window. `None` when
    /// neither owns a layer; the narrowest window when both do. The native MTP
    /// `auto` draft cap reads this.
    pub(crate) fn native_mtp_cpu_window_rows(&self) -> Option<usize> {
        let d1 = self
            .cpu_deferred_experts
            .then_some(cpu_splice::NARROW_VERIFY_MAX_ROWS);
        let d2 =
            (self.cpu_split_layers > 0).then_some(hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS);
        match (d1, d2) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    /// Trunk layers armed for the D2 split (any owner kind), in layer order.
    pub(crate) fn split_layers(&self) -> Vec<usize> {
        self.moe
            .iter()
            .flatten()
            .filter(|layer| layer.cpu_split.is_some())
            .map(|layer| layer.label)
            .collect()
    }
    /// The D2 owner kind of every armed split layer, in layer order.
    pub(crate) fn split_owner_kinds(&self) -> Vec<(usize, &'static str)> {
        self.moe
            .iter()
            .flatten()
            .filter(|layer| layer.cpu_split.is_some())
            .filter_map(|layer| {
                layer
                    .cpu_cold_owners
                    .as_ref()
                    .map(|owners| (layer.label, owners.kind().label()))
            })
            .collect()
    }
    /// The D2 cold-result return mode (`"mapped"` or `"copy"`) of every armed
    /// split layer, in layer order.
    pub(crate) fn split_return_modes(&self) -> Vec<(usize, &'static str)> {
        self.moe
            .iter()
            .flatten()
            .filter_map(|layer| {
                layer
                    .cpu_split
                    .as_ref()
                    .map(|executor| (layer.label, executor.return_mode().as_str()))
            })
            .collect()
    }
    /// The D2 input readiness mode (`"event"` or `"sync"`) of every armed
    /// split layer, in layer order.
    pub(crate) fn split_ready_modes(&self) -> Vec<(usize, &'static str)> {
        self.moe
            .iter()
            .flatten()
            .filter_map(|layer| {
                layer
                    .cpu_split
                    .as_ref()
                    .map(|executor| (layer.label, executor.ready_mode().as_str()))
            })
            .collect()
    }
    /// The D2 join mode (`"cpwait"` or `"host"`) of every armed split layer, in
    /// layer order.
    pub(crate) fn split_join_modes(&self) -> Vec<(usize, &'static str)> {
        self.moe
            .iter()
            .flatten()
            .filter_map(|layer| {
                layer
                    .cpu_split
                    .as_ref()
                    .map(|executor| (layer.label, executor.join_mode().as_str()))
            })
            .collect()
    }
    /// Cumulative D2 readiness/join counters of every armed split layer, as
    /// `(layer label, [ready_polls, ready_fallback_syncs, cpwait_joins,
    /// cpwait_aborts, watchdog_fires])` in layer order.
    pub(crate) fn split_readiness_counters(&self) -> Vec<(usize, [u64; 5])> {
        self.moe
            .iter()
            .flatten()
            .filter_map(|layer| {
                layer.cpu_split.as_ref().map(|executor| {
                    let (polls, fallbacks) = executor.ready_counters();
                    let (joins, aborts, fires) = executor.cpwait_counters();
                    (layer.label, [polls, fallbacks, joins, aborts, fires])
                })
            })
            .collect()
    }
    /// Layers armed for the D2 split through one owner kind.
    fn split_layers_of(&self, kind: cpu_splice::SplitOwnerKind) -> Vec<usize> {
        self.moe
            .iter()
            .flatten()
            .filter(|layer| {
                layer.cpu_split.is_some()
                    && layer
                        .cpu_cold_owners
                        .as_ref()
                        .is_some_and(|owners| owners.kind() == kind)
            })
            .map(|layer| layer.label)
            .collect()
    }
    /// The frozen split identity ([`Self::host_exec_identity`]) when any layer
    /// is armed for D2.
    pub(crate) fn split_digest(&self) -> Option<String> {
        (self.cpu_split_layers > 0).then(|| self.d2_exec_identity())
    }
    /// Numerical identity of the armed host arithmetic, for cache/session
    /// scoping. Empty when no CPU arithmetic is armed (default, `pcie`, or a
    /// requested CPU mode that fell back to PCIe on every layer), so the
    /// identity of every non-CPU load is unchanged.
    ///
    /// `cpu-all-v1/<bitmap>` names the D1 layers; `cpu-tier-split-v1/<tier
    /// logical digest>/<bitmap>/hotset-sha256=<sidecar sha | unbound>` the D2
    /// tier layers, `cpu-whole-split-v1/<bitmap>` the D2 un-tiered whole-layer
    /// host aggregates and `cpu-slab-split-v1/<bitmap>` the D2 host-mapped
    /// expert-slab arenas; the parts join with `+`. Bitmaps are hex
    /// with bit `i` = layer label `i` over the layers that actually armed.
    /// Thread count and scheduling are not part of it: they change timing,
    /// never arithmetic order.
    ///
    /// A load that admitted the SP short-prompt prefill hook (default off,
    /// `developer.qwen4_sp`) appends the part `sp-e1v1`: SP computes the CPU
    /// experts' rows with the E1 F32/BF16-boundary arithmetic, a different
    /// prefix-state domain, so an SP-on process never shares a snapshot (disk
    /// tier included) with an SP-off lineage.
    pub(crate) fn host_exec_identity(&self) -> String {
        crate::bundle::with_sp_identity(self.d2_exec_identity(), self.slowlink.sp_armed())
    }

    /// The D1/D2 part of [`Self::host_exec_identity`]: the frozen split
    /// identity ([`Self::split_digest`]) never carries the SP token.
    fn d2_exec_identity(&self) -> String {
        use cpu_splice::SplitOwnerKind::{Slab, Tier, WholeLayer};
        let cpu_all: Vec<usize> = self
            .moe
            .iter()
            .flatten()
            .filter(|layer| layer.cpu_deferred_experts)
            .map(|layer| layer.label)
            .collect();
        let tier_split = self.split_layers_of(Tier);
        let whole_split = self.split_layers_of(WholeLayer);
        let slab_split = self.split_layers_of(Slab);
        let tier = self
            .moe
            .iter()
            .flatten()
            .find(|layer| layer.cpu_split.is_some() && layer.tier.is_some())
            .and_then(|layer| layer.tier.as_ref())
            .map(|lease| {
                (
                    lease.logical_digest(),
                    lease.hotset_sidecar_sha256().map(str::to_owned),
                )
            });
        cpu_splice::host_exec_identity(
            &cpu_all,
            &[
                (Tier, &tier_split),
                (WholeLayer, &whole_split),
                (Slab, &slab_split),
            ],
            tier.as_ref()
                .map(|(digest, sidecar)| (digest, sidecar.as_deref())),
        )
    }
    /// Configure the D2 parity/timing hooks of every armed layer (lab only).
    pub(crate) fn split_parity_configure(
        &self,
        cfg: hipfire_dispatch::cpu_split::SplitParityConfig,
    ) -> Result<(), Qwen4GpuForwardError> {
        for layer in self.moe.iter().flatten() {
            if let Some(split) = layer.cpu_split.as_ref() {
                split.configure(layer.label, cfg.clone()).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!(
                        "layer {} split parity configure: {error:?}",
                        layer.label
                    ))
                })?;
            }
        }
        Ok(())
    }
    /// Drain every armed layer's parity records, ordered by (call, layer).
    pub(crate) fn split_take_parity_records(
        &self,
    ) -> Vec<hipfire_dispatch::cpu_split::SplitParityRecord> {
        let mut all = Vec::new();
        for layer in self.moe.iter().flatten() {
            if let Some(split) = layer.cpu_split.as_ref() {
                for (call, record) in split.take_parity_records().into_iter().enumerate() {
                    all.push((call, layer.label, record));
                }
            }
        }
        all.sort_by_key(|(call, label, _)| (*call, *label));
        all.into_iter().map(|(_, _, record)| record).collect()
    }
    pub(crate) fn ensure_cpu_splice_eager(
        &self,
        gpu: &Gpu,
        d2_call: bool,
    ) -> Result<(), Qwen4GpuForwardError> {
        // An automatic (default) D2 split asks for eager HIP only for the calls
        // that execute it (`d2_call`); explicit-mode and D1 loads keep the
        // whole-forward requirement.
        if !self.has_cpu_deferred_experts() && !(self.cpu_split_layers > 0 && d2_call) {
            return Ok(());
        }
        let stream_capture = match gpu.active_stream.as_ref() {
            Some(stream) => gpu.hip.stream_is_capturing(stream)?,
            None => false,
        };
        if gpu.graphs.capture_mode || gpu.replay.is_recording() || stream_capture {
            return Err(invalid(
                "qwen4 CPU routed experts require eager HIP; graph capture/recording cannot \
                 retain host-executed steps. Set replay.backend=hip or memory.offload_exec=pcie",
            ));
        }
        // The D2 tier split refuses every replay shape the executor's own gate
        // refuses (specialized verify/replay capture, retained bodies, AQL/PM4
        // routing, shadow bodies), here at admission rather than at the first
        // split call. D1 admission above is unchanged.
        if self.cpu_split_layers > 0
            && (gpu.graphs.verify.capturing.is_some()
                || gpu.graphs.replay.capturing.is_some()
                || gpu.replay.retained_body_active()
                || gpu.replay.should_route_aql()
                || gpu.replay.should_route_pm4()
                || gpu.replay.shadow_body_route().is_some()
                || gpu.replay.state() == rdna_compute::replay::ReplayState::ShadowValidated)
        {
            return Err(invalid(
                "qwen4 CPU tier split requires eager HIP; specialized capture, retained \
                 replay, AQL/PM4 and shadow bodies cannot retain the host slot arithmetic. \
                 Set replay.backend=hip or memory.offload_exec=pcie",
            ));
        }
        Ok(())
    }

    /// Refuse, before any allocation, a profile this forward cannot run: a
    /// full-attention layer needs the QSA RoPE kernels' hard-coded base.  PLE
    /// is optional: a profile without it has no PLE step, rows, staging,
    /// prefetch or lookahead in the forward.
    pub(crate) fn admit_profile(config: &Qwen4Config) -> Result<(), Qwen4GpuForwardError> {
        let id = config.profile.id();
        if config.layer_types.contains(&LayerType::FullAttention)
            && config.rope_theta != QWEN4_QSA_KERNEL_ROPE_THETA
        {
            return Err(invalid(format!(
                "profile {id} rope_theta {} differs from the {} base hard-coded in the QSA RoPE kernels",
                config.rope_theta, QWEN4_QSA_KERNEL_ROPE_THETA
            )));
        }
        Ok(())
    }

    pub fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::new_inner(
            gpu,
            bundle,
            max_chunk,
            mode,
            #[cfg(target_os = "linux")]
            None,
        )
    }

    /// Prepare the stock MQ6 trunk before the bundle swaps any overlay tensors.
    #[cfg(target_os = "linux")]
    pub(crate) fn prepare_npu_trunk(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        pre_overlay: bool,
        rows: usize,
        mode: Qwen4ForwardScratchMode,
    ) -> Result<Option<npu_forward::PreparedTrunk>, Qwen4GpuForwardError> {
        npu_forward::NpuForward::prepare_stock(gpu, bundle, pre_overlay, rows, mode)
    }

    /// Register the weight shadows of a decode-share load as non-owning views
    /// into the donor tail's shadow region ([`crate::decode_share_shadows`]):
    /// no allocation and no conversion here (every epoch's first acquire
    /// rebuilds them in place), and from now on neither shadow cache can ever
    /// allocate a shadow elsewhere on this `Gpu`. The region is re-planned
    /// from the manifest with the loader's flags and must equal the bytes the
    /// plan reserved.
    fn install_borrowed_shadows(
        gpu: &mut Gpu,
        weights: &crate::weights::Qwen4Weights,
        config: &Qwen4Config,
        max_chunk: usize,
        ranges: &crate::expert_tier_owners::DecodeShareRanges,
    ) -> Result<(), Qwen4GpuForwardError> {
        use rdna_compute::shadow_borrow::{F16ShadowSpan, ShadowBorrowPlan, SharedI8ShadowSpan};
        let (f16, shared_i8) = crate::decode_share_shadows::shadow_flags(
            &gpu.arch,
            gpu.qwen4_f16_shadow_capable(),
            gpu.qwen4_shared_i8_enabled(),
            max_chunk,
        )
        .map_err(|reason| invalid(format!("decode-share borrowed shadows: {reason}")))?;
        let region = crate::decode_share_shadows::plan_shadow_region(
            &weights.manifest.weights,
            config.num_hidden_layers,
            f16,
            shared_i8,
        )
        .map_err(invalid)?;
        if usize::try_from(region.total_bytes).ok() != Some(ranges.shadow_bytes) {
            return Err(invalid(format!(
                "decode-share borrowed shadows: the plan reserved {} B of donor tail but the \
                 manifest's shadow region is {} B (F16 {f16}, INT8 {shared_i8})",
                ranges.shadow_bytes, region.total_bytes
            )));
        }
        let base = ranges.donor_device.0;
        let at = |offset: u64| -> Result<usize, Qwen4GpuForwardError> {
            usize::try_from(offset)
                .ok()
                .and_then(|offset| base.checked_add(offset))
                .ok_or_else(|| invalid("decode-share shadow address overflows"))
        };
        let resident = |name: &str, layer: Option<usize>| {
            weights
                .resident_by_name(name, layer)
                .map_err(|error| invalid(format!("decode-share shadow source '{name}': {error}")))
        };
        let mut f16_spans = Vec::with_capacity(region.f16.len());
        for member in &region.f16 {
            let tensor = resident(&member.name, member.layer)?;
            if tensor.dtype != DType::BF16 || tensor.numel() != member.rows * member.cols {
                return Err(invalid(format!(
                    "decode-share shadow source '{}' is not its planned BF16 [{}, {}] matrix",
                    member.name, member.rows, member.cols
                )));
            }
            f16_spans.push(F16ShadowSpan {
                name: member.name.clone(),
                src_ptr: tensor.buf.as_ptr() as usize,
                dst_ptr: at(member.offset)?,
                rows: member.rows,
                cols: member.cols,
            });
        }
        let mut i8_spans = Vec::with_capacity(region.shared_i8.len());
        for member in &region.shared_i8 {
            let name = |proj: &str| {
                format!(
                    "model.language_model.layers.{}.mlp.shared_expert.{proj}.weight",
                    member.layer
                )
            };
            let ptr = |proj: &str| -> Result<usize, Qwen4GpuForwardError> {
                Ok(resident(&name(proj), Some(member.layer))?.buf.as_ptr() as usize)
            };
            i8_spans.push(SharedI8ShadowSpan {
                layer: member.layer,
                gate_ptr: ptr("gate_proj")?,
                up_ptr: ptr("up_proj")?,
                down_ptr: ptr("down_proj")?,
                intermediate: member.intermediate,
                hidden: member.hidden,
                down_rows: member.down_rows,
                gate_up_dst: at(member.gate_up_offset)?,
                down_dst: at(member.down_offset)?,
            });
        }
        gpu.install_shadow_borrow(ShadowBorrowPlan {
            region_ptr: base,
            region_bytes: ranges.shadow_bytes,
            f16: f16_spans,
            shared_i8: i8_spans,
        })?;
        eprintln!(
            "  qwen4 decode-share: {} F16 + {} INT8 weight shadows ({} MiB) live in the donor tail \
             (rebuilt per borrow epoch)",
            region.f16.len(),
            region.shared_i8.len(),
            ranges.shadow_bytes >> 20
        );
        Ok(())
    }

    /// Attach the donor-tail loan of a `fixed` / `on` load: claim the lease,
    /// check the largest class fits the donor span, install the borrowed
    /// weight shadows (when the plan reserved a shadow region), ensure the
    /// compute stream exists, and create the restore completion event. Writes
    /// nothing into the donor tail.
    fn attach_donor_loan(
        gpu: &mut Gpu,
        lease: &crate::expert_tier_owners::ExpertTierLease,
        span_bytes: u64,
        weights: &crate::weights::Qwen4Weights,
        config: &Qwen4Config,
        max_chunk: usize,
    ) -> Result<(DecodeShareBacking, DonorLoan), Qwen4GpuForwardError> {
        let ranges = lease
            .decode_share()
            .ok_or_else(|| invalid("the tier lease carries no decode-share donor tail"))?;
        let (base, bytes) = ranges.donor_device;
        let (mirror_cpu, mirror_bytes) = ranges.mirror_cpu;
        if span_bytes > bytes as u64 || bytes > mirror_bytes {
            return Err(invalid(format!(
                "the largest decode-share scratch class needs {span_bytes} bytes but the donor \
                 tail holds {bytes} (mirror {mirror_bytes})"
            )));
        }
        let claim = lease.claim_decode_share().map_err(invalid)?;
        gpu.ensure_capture_stream()?;
        let shadows = ranges.shadow_bytes > 0;
        if shadows {
            Self::install_borrowed_shadows(gpu, weights, config, max_chunk, &ranges)?;
        }
        let events = (|| -> Result<_, hip_bridge::HipError> {
            let done = gpu
                .hip
                .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?;
            // Timing events around the rebuild, only for the epoch log.
            let rebuild = if shadows && crate::decode_share::decode_share_log_enabled() {
                match (gpu.hip.event_create(), gpu.hip.event_create()) {
                    (Ok(start), Ok(end)) => Some((start, end)),
                    (Err(error), _) | (_, Err(error)) => {
                        let _ = gpu.hip.event_destroy(done);
                        return Err(error);
                    }
                }
            } else {
                None
            };
            Ok((done, rebuild))
        })();
        let (done, rebuild) = match events {
            Ok(events) => events,
            Err(error) => {
                if shadows {
                    gpu.clear_shadow_borrow();
                }
                return Err(error.into());
            }
        };
        Ok((
            DecodeShareBacking::Donor { base, bytes },
            DonorLoan {
                controller: crate::decode_share_loan::LoanController::new(
                    crate::decode_share::decode_share_sync_enabled(),
                    crate::decode_share::decode_share_log_enabled(),
                )
                .with_shadow_bytes(ranges.shadow_bytes as u64),
                claim,
                hw: LoanHw { done, rebuild },
                base,
                bytes,
                shadows,
                shadow_oracle_done: std::cell::Cell::new(false),
                mirror_cpu,
                verify: crate::decode_share::decode_share_verify_enabled(),
            },
        ))
    }

    /// The scratch of a decode-share load: the resident bank, one borrowed
    /// prefill bank per [`qwen4_decode_share_classes`] class, and the
    /// persistent token ids every bank views. The banks view either an owned
    /// test backing (`bank` mode, `donor` is `None`; charged nowhere) or the
    /// real donor tail of `donor` (`fixed` / `on`), borrowed under the loan
    /// controller. Returns the resident bank, the share state and the largest
    /// host token/PLE staging vectors.
    fn build_decode_share(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
        donor: Option<&crate::expert_tier_owners::ExpertTierLease>,
        weights: &crate::weights::Qwen4Weights,
    ) -> Result<(Qwen4GpuForwardScratch, DecodeShareBanks, Vec<u8>, Vec<u8>), Qwen4GpuForwardError>
    {
        let classes = qwen4_decode_share_classes(max_chunk);
        let ids_bytes = max_chunk
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| invalid("decode-share token id bytes overflow"))?;
        let token_ids = gpu.zeros(&[ids_bytes], DType::Raw)?;
        let (resident, _, mut host_ple) = match Qwen4GpuForwardScratch::new_resident_scoped(
            gpu,
            config,
            QWEN4_DECODE_SHARE_RESIDENT_ROWS,
            mode,
            scope,
            &token_ids,
        ) {
            Ok(built) => built,
            Err(error) => {
                let _ = gpu.free_tensor(token_ids);
                return Err(error);
            }
        };
        // The borrowed weight-shadow region (donor tail start; 0 over an owned
        // test backing or when the shadows stay resident): every class dirties it.
        let shadow_bytes = donor
            .and_then(|lease| lease.decode_share())
            .map_or(0u64, |ranges| ranges.shadow_bytes as u64);
        let backings = classes
            .iter()
            .map(|&rows| Qwen4GpuForwardScratch::borrowed_backing(config, rows, mode, scope))
            .collect::<Result<Vec<_>, _>>();
        let class_spans: Vec<u64> = match backings.and_then(|backings| {
            backings
                .iter()
                .map(|backing| {
                    backing
                        .span_bytes
                        .checked_add(shadow_bytes)
                        .ok_or_else(|| invalid("decode-share class span overflows"))
                })
                .collect::<Result<Vec<_>, _>>()
        }) {
            Ok(spans) => spans,
            Err(error) => {
                let _ = resident.free_gpu(gpu);
                let _ = gpu.free_tensor(token_ids);
                return Err(error);
            }
        };
        let span_bytes = class_spans.iter().copied().max().unwrap_or(0);
        let backing = match donor {
            Some(lease) => {
                Self::attach_donor_loan(gpu, lease, span_bytes, weights, config, max_chunk)
                    .map(|(backing, loan)| (backing, Some(loan)))
            }
            None => usize::try_from(span_bytes)
                .map_err(|_| invalid("decode-share backing bytes overflow"))
                .and_then(|span| gpu.zeros(&[span], DType::Raw).map_err(Into::into))
                .map(|tensor| (DecodeShareBacking::Owned(tensor), None)),
        };
        let (backing, loan) = match backing {
            Ok(pair) => pair,
            Err(error) => {
                let _ = resident.free_gpu(gpu);
                let _ = gpu.free_tensor(token_ids);
                return Err(error);
            }
        };
        let mut share = DecodeShareBanks {
            max_chunk,
            classes,
            class_spans,
            banks: Vec::new(),
            parked: None,
            active: None,
            token_ids,
            backing,
            loan,
            dirty_bytes: 0,
            stats: DecodeShareStats::default(),
            prompt_depth: 0,
            prompt_borrowed: false,
        };
        let mut host_token_bytes = vec![0u8; ids_bytes];
        // The scratch backing starts right after the borrowed weight shadows.
        let shadow = usize::try_from(shadow_bytes).unwrap_or(usize::MAX);
        let (base, span) = (
            share.backing.base().saturating_add(shadow),
            share.backing.bytes().saturating_sub(shadow),
        );
        let mut failure = None;
        for &rows in &share.classes.clone() {
            match Qwen4GpuForwardScratch::new_borrowed(
                config,
                rows,
                mode,
                scope,
                base,
                span,
                &share.token_ids,
            ) {
                Ok((bank, token_bytes, ple_bytes)) => {
                    if token_bytes.len() > host_token_bytes.len() {
                        host_token_bytes = token_bytes;
                    }
                    if ple_bytes.len() > host_ple.len() {
                        host_ple = ple_bytes;
                    }
                    share.banks.push(Some(bank));
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = failure {
            let _ = resident.free_gpu(gpu);
            let _ = share.free_gpu(gpu);
            return Err(error);
        }
        Ok((resident, share, host_token_bytes, host_ple))
    }

    pub(crate) fn new_inner(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        #[cfg(target_os = "linux")] npu_prepared: Option<Option<npu_forward::PreparedTrunk>>,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::new_inner_with_reserve(
            gpu,
            bundle,
            max_chunk,
            mode,
            #[cfg(target_os = "linux")]
            npu_prepared,
            None,
        )
    }

    /// [`Self::new_inner`] for an `HIPFIRE_EP_RESERVE_DECOUPLE=1` load when
    /// `ep_reserve_plan` carries the frozen plan: the 512-row resident bank plus
    /// the persistent 4096-row roots are the only scratch allocations, the 4096-row
    /// bank is views of donor memory, the loan mirrors the donors, and the borrowed
    /// compact runtimes and the one borrowed-only S2 stage are built over those
    /// mirrors. Any failed reserve verdict tears the whole forward down: there is
    /// no fallback to a full dedicated bank or a smaller chunk.
    pub(crate) fn new_inner_with_reserve(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_chunk: usize,
        mode: Qwen4ForwardScratchMode,
        #[cfg(target_os = "linux")] npu_prepared: Option<Option<npu_forward::PreparedTrunk>>,
        ep_reserve_plan: Option<Arc<crate::ep_reserve::EpReservePlan>>,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::admit_profile(&bundle.config)?;
        let scope = bundle.weights.band_scope().clone();
        let single_scope = scope == Qwen4BandScope::single(&bundle.config);
        check_band_admission(bundle, &scope, single_scope)?;
        let mesh_trunk_ops = mesh_trunk_op_range(&bundle.config, &scope)?;
        let flash_next = bundle.config.profile.is_flash_next();
        if let Some(plan) = ep_reserve_plan.as_ref() {
            if !flash_next
                || !single_scope
                || bundle.weights.ep_placement().is_none()
                || bundle.slowlink_profile().is_some()
                || mode != Qwen4ForwardScratchMode::Full
                || max_chunk != plan.forward.prompt_rows
            {
                return Err(invalid(format!(
                    "an EP reserve-sharing load needs the Flash-Next EP rank forward on the Full \
                     scratch plan at exactly the planned {}-row prompt chunk (got {max_chunk} rows, \
                     {mode:?})",
                    plan.forward.prompt_rows
                )));
            }
        }
        // Decode-share scratch banks: `bank` over an owned test backing,
        // `fixed` / `on` over the lease's real donor tail (only when the load
        // planned one).
        let share_mode = crate::decode_share::DecodeShareMode::from_env();
        let donor_lease = bundle
            .weights
            .expert_tiers()
            .filter(|lease| lease.decode_share().is_some());
        let want_banks = decode_share_banks_requested(share_mode)
            && (!share_mode.uses_donor_backing() || donor_lease.is_some());
        // Logged only for an explicit knob: the unset product default stays
        // silent on a load it does not apply to.
        let share_explicit = crate::decode_share::DecodeShareMode::explicit();
        if decode_share_banks_requested(share_mode) && !want_banks && share_explicit {
            eprintln!(
                "  qwen4 decode-share: {}={} not taken (the load carries no donor tail); \
                 ordinary single scratch bank",
                crate::decode_share::DECODE_SHARE_ENV,
                share_mode.name()
            );
        }
        let share_classes = if want_banks {
            let refusal = decode_share_refusal(
                &gpu.arch,
                mode,
                max_chunk,
                single_scope,
                bundle.weights.ep_placement().is_some() || bundle.ep.is_some(),
                bundle.pp.is_some() || bundle.pp_band.is_some(),
                bundle.slowlink_profile().is_some(),
            );
            if let (Some(reason), true) = (refusal, share_explicit) {
                eprintln!(
                    "  qwen4 decode-share: {}={} not taken ({reason}); \
                     ordinary single scratch bank",
                    crate::decode_share::DECODE_SHARE_ENV,
                    share_mode.name()
                );
            }
            refusal.is_none()
        } else {
            false
        };
        let mut reserve_banks: Option<ep_reserve_scratch::EpReserveBanks> = None;
        let (scratch, host_token_bytes, host_ple_bytes, share) = if share_classes {
            let donor = donor_lease.filter(|_| share_mode.uses_donor_backing());
            let (scratch, share, host_token_bytes, host_ple_bytes) = Self::build_decode_share(
                gpu,
                &bundle.config,
                max_chunk,
                mode,
                &scope,
                donor,
                &bundle.weights,
            )?;
            eprintln!(
                "  qwen4 decode-share[{}]: {}-row resident bank + prefill banks {:?} over {} of {} \
                 MiB (prompt chunk {max_chunk})",
                share_mode.name(),
                scratch.max_chunk,
                share.classes,
                if donor.is_some() {
                    "the donor tail"
                } else {
                    "an owned test backing"
                },
                share.class_spans.iter().copied().max().unwrap_or(0) >> 20
            );
            (scratch, host_token_bytes, host_ple_bytes, Some(share))
        } else if let Some(plan) = ep_reserve_plan.as_ref() {
            // EP reserve: the 512-row resident bank plus the two persistent 4096-row roots
            // are the only device allocations; the 4096-row bank is views of donor memory.
            // A full dedicated bank is never allocated, not even transiently.
            let (segments, placements) =
                ep_reserve_forward::EpReserveForward::layout(bundle, plan)?;
            let (scratch, mut banks) = ep_reserve_scratch::EpReserveBanks::build(
                gpu,
                &bundle.config,
                mode,
                plan.forward.prompt_rows,
                &segments,
                &placements,
            )?;
            let (host_token_bytes, host_ple_bytes) = banks.take_host_staging();
            reserve_banks = Some(banks);
            (scratch, host_token_bytes, host_ple_bytes, None)
        } else {
            let (scratch, host_token_bytes, host_ple_bytes) =
                Qwen4GpuForwardScratch::new_scoped(gpu, &bundle.config, max_chunk, mode, &scope)?;
            (scratch, host_token_bytes, host_ple_bytes, None)
        };
        let reserve = match (ep_reserve_plan, reserve_banks) {
            (Some(plan), Some(banks)) => {
                match ep_reserve_forward::EpReserveForward::materialize(gpu, bundle, plan, banks) {
                    Ok(reserve) => Some(reserve),
                    Err(error) => {
                        let _ = scratch.free_gpu(gpu);
                        return Err(error);
                    }
                }
            }
            _ => None,
        };
        let mut moe: Vec<Option<Qwen4MoeLayerRuntime>> =
            (0..bundle.config.num_hidden_layers).map(|_| None).collect();
        // BorrowPrefill runtimes of a decode-share binding (empty otherwise):
        // the per-layer MoE runtimes a prompt tile of at least the resident
        // bank's rows reads ([`Self::tile_moe`]).
        let mut moe_borrow: Vec<Vec<Option<Qwen4MoeLayerRuntime>>> = Vec::new();
        let mut borrow_class_rows: Vec<usize> = Vec::new();
        let mut decode_q8 = Vec::new();
        let mut route_trace = None;
        let mut cut_layers: Option<CutLayers> = None;
        let mut expert_stage = None;
        let mut ep: Option<Qwen4ForwardEp> = None;
        let mut trunk_zba = Vec::new();
        let mut trunk_a4 = Vec::new();
        #[cfg(target_os = "linux")]
        let mut npu_trunk = None;
        let result = (|| {
            // The D2 default is decided per load, before any layer admission
            // reads it and before stage buffers take VRAM or host memory; it
            // never touches the process-wide host-exec mode.
            let d2_resolved =
                crate::d2_default::resolve(gpu, &bundle.weights, &bundle.config, single_scope);
            // A `fixed` / `on` load placed its donor tail already and the split
            // has no host route for donors: the measured default stays off for
            // this load instead of failing it (`bind` is a numerics control and
            // keeps today's behavior).
            let d2_default =
                if d2_resolved && share_mode.uses_donor_backing() && donor_lease.is_some() {
                    eprintln!("  qwen4 D2 default: off (decode-share load)");
                    false
                } else {
                    d2_resolved
                };
            if let Some(placement) = bundle.weights.ep_placement() {
                let seal = bundle.ep_seal().cloned().ok_or_else(|| {
                    invalid("an EP placement needs the load's seal (bind_ep_seal)")
                })?;
                let state = ep.insert(Qwen4ForwardEp::new(gpu, &bundle.config, placement, seal)?);
                populate_moe_runtimes(&mut moe, &bundle.weights.layer_refs, |layer, weights| {
                    Qwen4MoeLayerRuntime::from_moe_ep(
                        gpu,
                        &bundle.weights,
                        weights,
                        layer,
                        &bundle.config,
                        placement,
                        &state.dummy,
                        &state.seal,
                        layer,
                    )
                })?;
                if let Some(reserve) = reserve.as_ref() {
                    // EP reserve borrowed compact view (`moe_borrow[0]`): every layer as the
                    // EP decode runtime, except the scratch donors `[H, R)`, whose owned
                    // experts name the loan's pinned mirror while their device bytes are scratch.
                    let donors = reserve.plan.donor_first_layer..reserve.plan.resident_layers;
                    moe_borrow.push((0..bundle.config.num_hidden_layers).map(|_| None).collect());
                    populate_moe_runtimes(
                        moe_borrow.last_mut().expect("pushed above"),
                        &bundle.weights.layer_refs,
                        |layer, weights| {
                            if !donors.contains(&layer) {
                                return Qwen4MoeLayerRuntime::from_moe_ep(
                                    gpu,
                                    &bundle.weights,
                                    weights,
                                    layer,
                                    &bundle.config,
                                    placement,
                                    &state.dummy,
                                    &state.seal,
                                    layer,
                                );
                            }
                            let typed = |slot: crate::ep_reserve::EpDonorSlot,
                                         source: &TensorRef|
                             -> Result<GpuTensor, Qwen4GpuForwardError> {
                                let dtype = bundle.weights.resident(source)?.dtype;
                                let mirror = reserve.mirror(layer, slot).ok_or_else(|| {
                                    invalid(format!(
                                        "EP reserve loan has no mirror of layer {layer} {slot:?}"
                                    ))
                                })?;
                                Ok(ep_reserve_forward::typed_mirror_source(&mirror, dtype))
                            };
                            let gate_up = typed(
                                crate::ep_reserve::EpDonorSlot::GateUp,
                                &weights.experts_gate_up,
                            )?;
                            let down =
                                typed(crate::ep_reserve::EpDonorSlot::Down, &weights.experts_down)?;
                            Qwen4MoeLayerRuntime::from_moe_ep_mirror(
                                gpu,
                                &bundle.weights,
                                weights,
                                layer,
                                &bundle.config,
                                placement,
                                &state.dummy,
                                &state.seal,
                                layer,
                                &gate_up,
                                &down,
                            )
                        },
                    )?;
                }
            } else {
                // Initial ctor keeps the default 4 until the real need is known;
                // the loader's post-admission prewarm (`reserve_d2_wide_rows`)
                // sizes armed layers to the actual need once, before the first
                // window (no hot growth, no in-flight replacement).
                populate_moe_runtimes(&mut moe, &bundle.weights.layer_refs, |layer, weights| {
                    Qwen4MoeLayerRuntime::from_moe(
                        gpu,
                        &bundle.weights,
                        weights,
                        layer,
                        &bundle.config,
                        d2_default,
                    )
                })?;
                if bundle
                    .weights
                    .expert_tiers()
                    .is_some_and(|lease| lease.decode_share().is_some())
                    && matches!(
                        crate::decode_share::DecodeShareMode::from_env(),
                        crate::decode_share::DecodeShareMode::Bind
                            | crate::decode_share::DecodeShareMode::Fixed
                            | crate::decode_share::DecodeShareMode::On
                    )
                {
                    let lease = bundle
                        .weights
                        .expert_tiers()
                        .expect("a decode-share binding implies a tier lease");
                    borrow_class_rows = qwen4_decode_share_classes(max_chunk);
                    // One view per prefill class (the loader sized them from the
                    // same chunk), or the single all-donors view of a plan whose
                    // classes were never set.
                    let views = lease.borrow_classes();
                    if views != 1 && views != borrow_class_rows.len() {
                        return Err(invalid(format!(
                            "decode-share: the lease carries {views} BorrowPrefill class views but \
                             the {max_chunk}-row chunk has {} prefill classes",
                            borrow_class_rows.len()
                        )));
                    }
                    for class in 0..views {
                        // Owned by the forward before it is populated: an error
                        // below frees whatever was built.
                        moe_borrow
                            .push((0..bundle.config.num_hidden_layers).map(|_| None).collect());
                        populate_moe_runtimes(
                            moe_borrow.last_mut().expect("pushed above"),
                            &bundle.weights.layer_refs,
                            |layer, weights| {
                                Qwen4MoeLayerRuntime::from_moe_borrow(
                                    gpu,
                                    &bundle.weights,
                                    weights,
                                    layer,
                                    &bundle.config,
                                    d2_default,
                                    class,
                                )
                            },
                        )?;
                    }
                    Self::check_borrow_runtimes(
                        bundle,
                        &moe,
                        &moe_borrow,
                        share.as_ref().map(|share| share.class_spans.as_slice()),
                    )?;
                }
            }
            let moe_count = moe.iter().flatten().count();
            // A band keeps every routed expert resident (packed aggregates or
            // its own layers' slab arenas): no host-mapped layer, no stage.
            if !single_scope && moe.iter().flatten().any(|layer| layer.host_mapped()) {
                return Err(invalid(
                    "a pipeline-parallel band refuses host-mapped expert layers",
                ));
            }
            let cpu_layers = moe
                .iter()
                .flatten()
                .filter(|layer| layer.cpu_deferred_experts)
                .count();
            // D2 split: a distinct numerical mode, per owner kind.
            let split_layers = moe
                .iter()
                .flatten()
                .filter(|layer| layer.cpu_split.is_some())
                .count();
            let tier_split_layers = moe
                .iter()
                .flatten()
                .filter(|layer| layer.cpu_split.is_some() && layer.tier.is_some())
                .count();
            let tier_layers = moe
                .iter()
                .flatten()
                .filter(|layer| layer.tier.is_some())
                .count();
            if (hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled()
                && !hipfire_dispatch::cpu_exec::passback_enabled()
                || d2_default)
                && tier_layers > 0
            {
                let cold: usize = moe
                    .iter()
                    .flatten()
                    .filter_map(|layer| layer.cpu_cold_owners.as_ref())
                    .filter(|owners| owners.kind() == cpu_splice::SplitOwnerKind::Tier)
                    .map(cpu_splice::HostSpanOwner::cold_experts)
                    .sum();
                let slots = tier_split_layers * bundle.config.num_experts;
                // The same frozen plan hash scopes host_exec_identity/split_digest.
                // Report the assignment, not merely its hot/cold slot counts.
                let partition_digest = moe
                    .iter()
                    .flatten()
                    .find(|layer| layer.cpu_split.is_some() && layer.tier.is_some())
                    .and_then(|layer| layer.tier.as_ref())
                    .map(|lease| {
                        let mut hex = String::with_capacity(64);
                        for byte in lease.logical_digest() {
                            use std::fmt::Write as _;
                            let _ = write!(hex, "{byte:02x}");
                        }
                        hex
                    })
                    .unwrap_or_else(|| "unarmed".to_owned());
                eprintln!(
                    "  [moe-cpu-split] qwen4: {tier_split_layers}/{tier_layers} hot-set layers armed for the CPU tier split (static slot placement across armed layers: {} hot / {cold} cold expert slots of {slots}; dynamic per-token route hot/cold counts are reported by the executor); {} CPU worker threads; eager HIP for decode and <=4-row verify and replay, wide prefill keeps PCIe; layers whose host spans could not be resolved stay on PCIe; partition_digest={partition_digest}",
                    slots - cold,
                    hipfire_dispatch::cpu_exec::splice_pool_threads()
                );
            }
            // Un-tiered owner kinds: every host-placed expert of an armed
            // layer is cold.
            for (kind, what) in [
                (
                    cpu_splice::SplitOwnerKind::WholeLayer,
                    "whole-layer host aggregate",
                ),
                (cpu_splice::SplitOwnerKind::Slab, "host-mapped expert-slab"),
            ] {
                let labels: Vec<usize> = moe
                    .iter()
                    .flatten()
                    .filter(|layer| {
                        layer.cpu_split.is_some()
                            && layer
                                .cpu_cold_owners
                                .as_ref()
                                .is_some_and(|owners| owners.kind() == kind)
                    })
                    .map(|layer| layer.label)
                    .collect();
                if labels.is_empty() {
                    continue;
                }
                let cold: usize = moe
                    .iter()
                    .flatten()
                    .filter_map(|layer| layer.cpu_cold_owners.as_ref())
                    .filter(|owners| owners.kind() == kind)
                    .map(cpu_splice::HostSpanOwner::cold_experts)
                    .sum();
                eprintln!(
                    "  [moe-cpu-split] qwen4: {} {what} layers armed for the CPU split ({} layer bitmap {}; every host-placed expert is cold: {cold} expert slots; decode and <=4-row verify of those experts run on the CPU); {} CPU worker threads; eager HIP for decode and <=4-row verify and replay, wide prefill keeps PCIe, placement and host allocations are unchanged; layers whose host spans could not be resolved stay on PCIe",
                    labels.len(),
                    kind.label(),
                    cpu_splice::layer_bitmap_hex(&labels),
                    hipfire_dispatch::cpu_exec::splice_pool_threads()
                );
                if kind == cpu_splice::SplitOwnerKind::Slab {
                    // Name the CPU kernel of every format the armed slab
                    // layers decode, so each run records its E8P/W3/MQ4 path.
                    let dtypes: std::collections::BTreeSet<DType> = moe
                        .iter()
                        .flatten()
                        .filter(|layer| layer.cpu_split.is_some())
                        .filter_map(|layer| match layer.cpu_cold_owners.as_ref() {
                            Some(cpu_splice::HostSpanOwner::Slab(owners)) => Some(owners),
                            _ => None,
                        })
                        .flat_map(|owners| owners.dtypes())
                        .collect();
                    let kernels: Vec<String> = dtypes
                        .into_iter()
                        .map(|dtype| {
                            let kernel =
                                hipfire_dispatch::cpu_exec::cpu_kernel_for(dtype).unwrap_or("none");
                            format!("{dtype:?}={kernel}")
                        })
                        .collect();
                    eprintln!(
                        "  [moe-cpu-split] qwen4: host-mapped expert-slab CPU kernels: {}",
                        kernels.join(" ")
                    );
                }
            }
            if hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled() {
                let slab_layers = moe
                    .iter()
                    .flatten()
                    .filter(|layer| layer.slab.is_some())
                    .count();
                eprintln!(
                    "  [moe-cpu-expert] qwen4: {cpu_layers}/{} trunk layers armed for CPU routed FFN (armed layers use eager HIP for decode and <=4-row verify; wide prefill keeps PCIe); {slab_layers} expert-slab layers never take this splice (a host-mapped slab arena uses the D2 split, see [moe-cpu-split]); EP, AWQ, unsupported dtypes and resident layers keep the GPU path; expert-tier/hot-set layers use the separate hot-set tier split (see [moe-cpu-split])",
                    moe_count
                );
            }
            if cpu_layers + split_layers > 0 {
                let stream_capture = match gpu.active_stream.as_ref() {
                    Some(stream) => gpu.hip.stream_is_capturing(stream)?,
                    None => false,
                };
                if gpu.graphs.capture_mode || gpu.replay.is_recording() || stream_capture {
                    return Err(invalid(
                        "qwen4 CPU routed experts cannot be attached during capture/recording: \
                         set replay.backend=hip or memory.offload_exec=pcie",
                    ));
                }
                if gpu.replay.request() == rdna_compute::replay::ReplayBackendRequest::Shadow {
                    return Err(invalid(
                        "qwen4 CPU routed experts conflict with replay.backend=shadow: \
                         retained replay cannot execute the host splice. Set replay.backend=hip \
                         or memory.offload_exec=pcie",
                    ));
                }
                // A CPU splice is a host synchronization boundary, absent from
                // a retained tape. Auto replay is disabled before publication.
                gpu.replay.poison(
                    "qwen4 CPU routed experts require eager HIP (memory.offload_exec=cpu|passback or the measured D2 default)",
                );
                hipfire_dispatch::cpu_exec::log_capture_disabled_once();
            }
            let compact = mode == Qwen4ForwardScratchMode::FnSymGfx1201;
            // Every MoE layer must run one route the compact buffers were
            // sized for: all expert-slab layers (K1 lane-union GEMVs inside
            // the 512-row F32 fallback, K2 grouped GEMMs writing the BF16
            // grouped rows) or all symmetric-IU4 layers.
            let slab_layers = moe
                .iter()
                .flatten()
                .filter(|layer| layer.slab.is_some())
                .count();
            let all_slab = moe_count != 0 && slab_layers == moe_count;
            if compact && slab_layers != 0 && !all_slab {
                return Err(invalid(
                    "compact scratch needs every MoE layer on one route (expert-slab arenas or \
                     symmetric IU4 headers), but only some layers are expert-slab arenas",
                ));
            }
            if compact {
                // The compact plan was admitted before any weight was
                // loaded; it is valid only for the exact route it sized.
                if gpu.arch != "gfx1201" || !gpu.flags.moe_grouped_gemm {
                    return Err(invalid(
                        "compact scratch requires gfx1201 with the grouped MoE GEMM",
                    ));
                }
                if !all_slab && !gpu.qwen4_moe_sym_iu4_requested() {
                    return Err(invalid(
                        "compact scratch requires gfx1201 with the grouped MoE GEMM and the \
                         symmetric IU4 route requested",
                    ));
                }
                for layer in &bundle.weights.layer_refs {
                    if let Some(ple) = layer.ple.as_ref() {
                        let ple = ple_desc(&bundle.weights, ple)?;
                        if ple.key.dtype != DType::BF16 || ple.value.dtype != DType::BF16 {
                            return Err(invalid(
                                "compact scratch requires BF16 PLE key/value projections",
                            ));
                        }
                    }
                }
            }
            // Symmetric IU4 MoE (`HIPFIRE_QWEN4_MOE_SYM_IU4`: default on for gfx1100, gfx1151
            // and gfx1201, `=0` off): verify each
            // layer's final expert headers once, before any capture, and size the route's
            // scratch for the chunk cap. Layers that fail (every layer of an
            // asymmetric artifact) keep the F16 route; nothing runs (and no new
            // module loads) unless the route was requested. The loop covers EVERY routed trunk
            // layer, the S1 host-mapped ones included: their pointer tables hold the
            // device-visible aliases of the same pinned payload the S1 transport pages DMA from
            // ([`Qwen4MoeLayerRuntime::s1_host_sources`] translates those very entries).
            // (An EP rank verifies its full chunk tables in `attach_ep`, once
            // every peer's experts are mapped.)
            // Owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`): its grouped-row
            // buffer, masked tile ids and sym scratch exist before the mesh enables peers.
            let exchange = reserve
                .as_ref()
                .map(|reserve| reserve.banks.exchange_capacity());
            if let Some(state) = ep.as_mut() {
                if qwen4_ep_owner_routed_prefill_enabled() {
                    state.tile = Qwen4TileRoute::admit(
                        gpu,
                        &bundle.config,
                        state.placement,
                        &state.seal.seal.owner_ranks(),
                        &scratch,
                        mode,
                        max_chunk,
                        flash_next,
                        exchange.as_ref(),
                    )?;
                }
            }
            // Stock FP8: no QT44/QT53 headers to verify; the grouped route runs
            // in the Full plan's preallocated F32 buffers, proved here once.
            let stock_fp8 = bundle.weights.manifest.identity.recipe.is_stock_fp8();
            if stock_fp8 && moe_count > 0 {
                check_fp8_grouped_scratch(&scratch, &bundle.config, max_chunk)?;
                // The stable-scatter grouping scratch is reserved here, before any
                // forward or capture, so no first decode/prefill grows it per call.
                gpu.try_gfx942()
                    .ok_or_else(|| invalid("the qt=55 FP8 route needs a gfx942 device"))?
                    .reserve_fp8b128_moe_group(
                        max_chunk,
                        bundle.config.num_experts_per_tok,
                        bundle.config.num_experts,
                    )?;
            }
            // An S1 load whose window route resolved to the F16 `Aggregate` lowering (flag off,
            // or requested but a routed layer's source headers / recipe refused it) is
            // whole-profile F16: no layer verifies symmetric, so no tile anywhere classifies as
            // IU4 and no uncharged sym scratch is reserved (mixed verdicts are not budgeted).
            let s1_f16_profile = bundle.slowlink_profile().is_some_and(|profile| {
                profile.geometry.route == crate::slowlink_admission::WindowRoute::Aggregate
            });
            if !stock_fp8
                && ep.is_none()
                && moe_count > 0
                && !s1_f16_profile
                && gpu.qwen4_moe_sym_iu4_requested()
            {
                let mut verified = 0usize;
                for layer in moe.iter_mut().flatten() {
                    verified += usize::from(layer.verify_symmetric(gpu, &bundle.config)?);
                }
                eprintln!(
                    "  qwen4 symmetric IU4 MoE: {verified}/{} layers verified symmetric",
                    moe_count
                );
                // The compact scratch has no buffer for the F32 route of a
                // layer that is not symmetric: never run it silently.  An
                // all-slab load is not on the symmetric route (its K2 GEMMs
                // write the same grouped-row buffers, K1 stays inside the
                // 512-row F32 fallback), so it has no symmetric count to meet.
                // Developer knob: refuse the compact plan as if one layer were
                // unverified, without altering any layer's verdict or route.
                let accepted =
                    if hipfire_config::developer_bool(QWEN4_COMPACT_FORCE_UNVERIFIED_ENV, false) {
                        verified.saturating_sub(1)
                    } else {
                        verified
                    };
                if compact && !all_slab && accepted != moe_count {
                    return Err(Qwen4GpuForwardError::CompactUnverified {
                        verified: accepted,
                        total: moe_count,
                    });
                }
                if verified > 0 {
                    gpu.reserve_qwen4_moe_sym(
                        max_chunk,
                        bundle.config.num_experts_per_tok,
                        bundle.config.num_experts,
                        bundle.config.hidden_size,
                        bundle.config.moe_intermediate_size,
                    )?;
                }
            }
            // Sealed expert-slab layers on gfx1100: the symmetric IU4 block above does not run
            // there when `HIPFIRE_QWEN4_MOE_SYM_IU4=0` (or no layer passed the header check), yet
            // the slab K2 prefill reuses its grouping scratch, down sidecar and shared gate/up
            // scratch. Reserve them here, before any capture, through the same reserve (no
            // growth when that block already ran). gfx1201 slab loads are unchanged.
            if !stock_fp8
                && ep.is_none()
                && gpu.arch == "gfx1100"
                && moe.iter().flatten().any(|layer| layer.slab.is_some())
            {
                gpu.reserve_qwen4_moe_sym(
                    max_chunk,
                    bundle.config.num_experts_per_tok,
                    bundle.config.num_experts,
                    bundle.config.hidden_size,
                    bundle.config.moe_intermediate_size,
                )?;
            }
            // Phase 2 of the symmetric IU4 window route proof (the source-header scan before
            // placement was phase 1): a load charged for `AggregateSymIu4` needs every routed
            // trunk layer's FINAL bound tables verified above. A shortfall fails the load here,
            // before serving, never a silent switch to a differently budgeted route.
            if let Some(route) = bundle
                .slowlink_profile()
                .map(|profile| profile.geometry.route)
            {
                let total = moe.iter().flatten().count();
                let verified = moe.iter().flatten().filter(|layer| layer.symmetric).count();
                crate::slowlink_admission::check_sym_iu4_final_headers(route, verified, total)
                    .map_err(invalid)?;
            }
            // A BorrowPrefill runtime inherits its Resident twin's symmetric
            // verdict: `verify_symmetric` proved the IU4 header grid on payloads
            // the donors' mirror reproduces byte for byte
            // (`ExpertTierLease::verify_donor_tail` under the verify knob), so
            // the second device check would only re-read the same bytes.
            for runtimes in moe_borrow.iter_mut() {
                for (borrowed, resident) in runtimes.iter_mut().zip(&moe) {
                    if let (Some(borrowed), Some(resident)) = (borrowed, resident) {
                        borrowed.symmetric = resident.symmetric;
                    }
                }
            }
            if flash_next {
                (trunk_zba, trunk_a4) = prepare_trunk_iu4(gpu, bundle)?;
                if ep.is_none()
                    || (ep_lever_on(
                        hipfire_config::developer_var(Qwen4ExpertStage::EP_ENV)
                            .ok()
                            .as_deref(),
                    ) && ep.as_ref().is_some_and(|state| state.tile.is_some()))
                {
                    // An S1 load charged its own page ring instead of the G2
                    // stage (`PlacementCharges::slowlink` with `staged = false`):
                    // no donor or slab stage buffer is created beside it, and a
                    // request S1 declines reads the host-mapped experts in place.
                    if single_scope && bundle.slowlink_profile().is_none() {
                        expert_stage = match (reserve.as_ref(), moe_borrow.first()) {
                            // EP reserve: the one borrowed-only S2 configuration over the borrowed
                            // view (first host layer H, donors H-2 / H-1, loan-mirror host
                            // sources for [H, R), the original pinned owners for [R, 48)).
                            (Some(reserve), Some(view)) => Qwen4ExpertStage::new(
                                gpu,
                                bundle,
                                view,
                                max_chunk,
                                Some(&reserve.host_sources),
                            )?,
                            _ => Qwen4ExpertStage::new(gpu, bundle, &moe, max_chunk, None)?,
                        };
                    }
                }
            } else {
                gpu.qwen4_trunk_iu4 = false;
            }
            if let Some(reserve) = reserve.as_ref() {
                // Phase two of the reserve admission: the GPU/runtime verdicts a reduced-reserve
                // load needs on this rank. A shortfall tears the load down; nothing falls back to
                // the ordinary path with the reduced reserve already spent.
                let s1 = ep.as_ref().is_some_and(|state| state.tile.is_some());
                let Some(stage) = expert_stage.as_ref() else {
                    return Err(invalid(
                        "EP reserve sharing needs the borrowed-view expert stage (S2) and it was \
                         not admitted on this rank (needs HIPFIRE_EP_EXPERT_STAGE on, gfx1201, a \
                         pinned host source for every host layer)",
                    ));
                };
                if !s1 {
                    return Err(invalid(
                        "EP reserve sharing needs owner-routed prefill (S1) and it was not admitted \
                         on this rank (HIPFIRE_EP_OWNER_ROUTED_PREFILL, the symmetric IU4 route)",
                    ));
                }
                if stage.first_host != reserve.plan.donor_first_layer
                    || stage.donors != reserve.plan.s2_donor_layers
                    || stage.min_rows <= reserve.banks.resident_rows()
                {
                    return Err(invalid(format!(
                        "EP reserve stage mismatch: first host layer {} / donors {:?} / min rows {} \
                         against the plan's {} / {:?} (a borrowed-only stage must not serve the \
                         {}-row resident bank)",
                        stage.first_host,
                        stage.donors,
                        stage.min_rows,
                        reserve.plan.donor_first_layer,
                        reserve.plan.s2_donor_layers,
                        reserve.banks.resident_rows()
                    )));
                }
            }
            #[cfg(target_os = "linux")]
            {
                npu_trunk = npu_forward::NpuForward::load(
                    gpu,
                    bundle,
                    &scratch,
                    max_chunk,
                    mode,
                    npu_prepared,
                )?;
            }
            // Stock-FP8 always reads source BF16 for decode and prefill.
            // Q8 copies remain the gfx1151 packed Flash-Next bandwidth recipe.
            let mut source_refs = Vec::new();
            if flash_next && !bundle.weights.manifest.identity.recipe.is_stock_fp8() {
                for layer in &bundle.weights.layer_refs {
                    let moe = layer
                        .ffn
                        .moe()
                        .ok_or_else(|| invalid("Flash-Next layer without an MoE FFN"))?;
                    source_refs.extend([
                        &layer.attn_hyper.input_mix_down,
                        &layer.attn_hyper.input_mix_up,
                        &layer.mlp_hyper.input_mix_down,
                        &layer.mlp_hyper.input_mix_up,
                        &moe.shared_gate,
                        &moe.shared_up,
                        &moe.shared_down,
                    ]);
                }
            }
            let sources = source_refs
                .into_iter()
                .map(|reference| dense_ref(&bundle.weights, reference))
                .collect::<Result<Vec<_>, _>>()?;
            if gpu.arch_caps.is_gfx1151()
                && sources
                    .iter()
                    .all(|weight| weight.dtype == DType::BF16 && weight.k % 32 == 0)
            {
                if !single_scope {
                    return Err(invalid(
                        "a pipeline-parallel band refuses the gfx1151 Q8 decode copies",
                    ));
                }
                for weight in sources {
                    decode_q8.push(gpu.quantize_bf16_q8_0(weight.buf, weight.m, weight.k)?);
                }
            }
            if let Some(path) = hipfire_config::developer_var(ROUTE_TRACE_ENV)
                .ok()
                .filter(|_| flash_next)
            {
                if !single_scope {
                    return Err(invalid(format!(
                        "{ROUTE_TRACE_ENV} has no meaning for a pipeline-parallel band"
                    )));
                }
                let slots = bundle.config.num_hidden_layers * bundle.config.num_experts_per_tok;
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|error| invalid(format!("{ROUTE_TRACE_ENV}={path}: {error}")))?;
                route_trace = Some(RouteTrace {
                    device: gpu.alloc_tensor(&[slots], DType::F32)?,
                    host: vec![0u8; slots * 4],
                    out: std::io::BufWriter::new(file),
                });
                eprintln!("  qwen4 routing trace -> {path}");
            }
            if let Ok(raw) = hipfire_config::developer_var(CUT_LAYERS_ENV) {
                if !single_scope {
                    return Err(invalid(format!(
                        "{CUT_LAYERS_ENV} cannot be set for a pipeline-parallel band (the band edge is the cut)"
                    )));
                }
                let config = &bundle.config;
                let ple_layer_index = config
                    .ple
                    .as_ref()
                    .and_then(|_| config.ple_layer_ids.first())
                    .and_then(|id| id.checked_sub(1));
                cut_layers = Some(
                    parse_qwen4_cut_layers(&raw, config.num_hidden_layers, ple_layer_index)
                        .map_err(|error| invalid(format!("{CUT_LAYERS_ENV}={raw}: {error}")))?,
                );
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            #[cfg(target_os = "linux")]
            if let Some(npu) = npu_trunk {
                let _ = npu.free(gpu);
            }
            if let Some(stage) = expert_stage {
                let _ = stage.free_gpu(gpu);
            }
            for layer in moe_borrow.into_iter().flatten().flatten() {
                let _ = layer.free_gpu(gpu);
            }
            for layer in moe.into_iter().flatten() {
                let _ = layer.free_gpu(gpu);
            }
            if let Some(ep) = ep {
                let _ = ep.free_gpu(gpu);
            }
            gpu.qwen4_trunk_iu4 = false;
            for tensor in decode_q8.into_iter().chain(trunk_zba.into_iter().flatten()) {
                let _ = gpu.free_tensor(tensor);
            }
            if let Some(trace) = route_trace {
                let _ = gpu.free_tensor(trace.device);
            }
            let _ = scratch.free_gpu(gpu);
            if let Some(share) = share {
                let _ = share.free_gpu(gpu);
            }
            if let Some(reserve) = reserve {
                let ep_reserve_forward::EpReserveTeardown { banks, loan } = reserve.teardown(gpu);
                let _ = banks.release(gpu);
                if let Some((loan, permit)) = loan {
                    let _ = loan.release(gpu, permit);
                }
            }
            return Err(error);
        }
        Ok(Self {
            expert_stage,
            ep,
            scratch,
            host_token_bytes,
            uploaded_id_rows: 0,
            host_ple: PleHostStage::new(
                host_ple_bytes.len(),
                ple_async_upload_enabled(
                    hipfire_config::developer_var(PLE_ASYNC_UPLOAD_ENV)
                        .ok()
                        .as_deref(),
                ),
            ),
            ple_lookahead: Vec::new(),
            ple_ahead: None,
            token_readback: None,
            token_readback_early: hipfire_config::developer_bool(
                "HIPFIRE_QWEN4_TOKEN_READBACK_EARLY",
                true,
            ),
            cpu_deferred_experts: moe.iter().flatten().any(|layer| layer.cpu_deferred_experts),
            cpu_split_layers: moe
                .iter()
                .flatten()
                .filter(|layer| layer.cpu_split.is_some())
                .count(),
            cpu_split_explicit: hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled()
                && !hipfire_dispatch::cpu_exec::passback_enabled(),
            moe,
            moe_borrow,
            borrow_class_rows,
            borrow_runtimes_active: false,
            borrow_view: 0,
            decode_q8,
            route_trace,
            cut_layers,
            trunk_zba,
            trunk_a4,
            qsa_tap: None,
            layer_capture: None,
            lane_scope: false,
            prefill_scope: false,
            #[cfg(feature = "lab")]
            last_verify_reference: None,
            scope,
            mesh_route_rows: None,
            mesh_segment: None,
            mesh_next_op: None,
            mesh_tokens: PleHostStage::new(max_chunk * std::mem::size_of::<i32>(), true),
            lane_tokens: crate::lane_token_ring::LaneTokenRing::new(
                max_chunk * std::mem::size_of::<i32>(),
            ),
            mesh_trunk_ops,
            slowlink: Default::default(),
            share,
            ep_ordinary_receipt: (reserve.is_none()
                && hipfire_config::developer_bool(
                    ep_reserve_forward::EP_RESERVE_RECEIPT_ENV,
                    false,
                ))
            .then(Default::default),
            receipt_mtp_fill: false,
            ep_reserve: reserve,
            #[cfg(target_os = "linux")]
            npu_trunk,
        })
    }

    /// Whether planned pairs run layer-major through the pair executor (`on`,
    /// `auto`, and `emulate` with a bank-B scratch): the capture capacity covers
    /// both banks of a group.
    pub(crate) fn npu_trunk_pipelined(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.npu_trunk.as_ref().is_some_and(|n| n.pipelined())
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Latched faults and diagnostic GPU-only requests take exactly the `off` scheduling.
    pub(crate) fn npu_trunk_planned(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.npu_trunk.as_ref().is_some_and(|n| !n.route_off())
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Whether the request in flight tripped the route's fault latch; clears the per-request flag.
    pub(crate) fn take_npu_trunk_fault(&mut self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.npu_trunk
                .as_mut()
                .is_some_and(|n| n.take_request_fault())
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    pub(crate) fn begin_npu_trunk_request(
        &mut self,
        gpu: &Gpu,
    ) -> Result<(), Qwen4GpuForwardError> {
        #[cfg(target_os = "linux")]
        if let Some(npu) = &mut self.npu_trunk {
            npu.begin_request(gpu)?;
        }
        #[cfg(not(target_os = "linux"))]
        let _ = gpu;
        Ok(())
    }

    /// [`Self::begin_npu_trunk_request`]'s request runs the GPU-only
    /// schedule: no NPU pair is planned or selected for it.
    pub(crate) fn npu_trunk_request_gpu_only(&mut self, why: &str) {
        #[cfg(target_os = "linux")]
        if let Some(npu) = &mut self.npu_trunk {
            npu.force_gpu_request(why);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = why;
    }

    /// The modes the NPU trunk pair executor (layer-major, outside the step
    /// interpreter) cannot run, checked before a selected pair runs.
    fn refuse_npu_pair(&self) -> Result<(), Qwen4GpuForwardError> {
        if self.cut_layers.is_some() {
            return Err(invalid(format!(
                "{CUT_LAYERS_ENV} cannot run with the NPU trunk pair route"
            )));
        }
        if self.layer_capture.is_some() {
            return Err(invalid(
                "the DFlash2 layer capture cannot run with the NPU trunk pair route \
                 (its prefill must be planned GPU-only)",
            ));
        }
        Ok(())
    }

    pub(crate) fn plan_npu_trunk_request(
        &mut self,
        gpu: &Gpu,
        rows: usize,
        position: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        #[cfg(target_os = "linux")]
        if let Some(npu) = &mut self.npu_trunk {
            npu.plan_request(gpu, rows, position)?;
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (gpu, rows, position);
        Ok(())
    }

    /// Advance the executor's pending completions; must precede every
    /// [`Self::npu_trunk_pair`] decision on a mutable path.
    pub(crate) fn poll_npu_trunk(&mut self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        #[cfg(target_os = "linux")]
        if let Some(npu) = &mut self.npu_trunk {
            npu.poll(gpu)?;
        }
        #[cfg(not(target_os = "linux"))]
        let _ = gpu;
        Ok(())
    }

    /// Rows of banks A and B of the pair the plan starts at `position`.
    pub(crate) fn npu_trunk_pair(
        &self,
        position: usize,
        remaining: usize,
        chunk: usize,
    ) -> Option<[usize; 2]> {
        #[cfg(target_os = "linux")]
        {
            self.npu_trunk
                .as_ref()
                .and_then(|n| n.selected_pair(position, remaining, chunk))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (position, remaining, chunk);
            None
        }
    }

    pub(crate) fn note_npu_trunk_pair(&mut self, rows: [usize; 2], position: usize) {
        #[cfg(target_os = "linux")]
        if let Some(npu) = &mut self.npu_trunk {
            npu.note_pair(rows, position);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (rows, position);
    }

    /// Before a forward that is not this owner's chunk program (the lane
    /// step) reads any expert: restore the donor layers a failed staged
    /// forward may have left overwritten. A no-op without G2 staging.
    pub(crate) fn restore_expert_stage(&self, gpu: &mut Gpu) -> Result<(), Qwen4GpuForwardError> {
        match self.expert_stage.as_ref() {
            Some(stage) => stage.restore_blocking(gpu),
            None => Ok(()),
        }
    }

    /// This rank's expert-parallel placement, `None` on the single-device route.
    pub(crate) fn ep_placement(&self) -> Option<Qwen4EpPlacement> {
        self.ep.as_ref().map(|ep| ep.placement)
    }

    /// How `layer`'s routed experts are bound by an EP lane step, `None` unless
    /// positively one of the two proven packed MQ4 (QT44/QT53) kinds: the
    /// layer's static resident runtime, or its live balanced runtime over a
    /// host-mapped trunk layer. Read-only and allocation-free: it inspects the
    /// runtime [`lane_layer_moe`] selects (the one selector the lane step
    /// binds), the installed row gather and the runtimes' plain flags.
    pub(crate) fn lane_ep_expert_kind(&self, layer: usize) -> Option<EpLaneExpertKind> {
        let ep = self.ep.as_ref()?;
        let selected = lane_layer_moe(Some(ep), &self.moe, layer)?;
        let balanced_selected = ep.balanced.get(layer).and_then(Option::as_ref).is_some();
        classify_ep_lane_expert(
            layer,
            ep.gather.is_some(),
            balanced_selected,
            Some(selected.ep_lane_facts()),
            self.moe
                .get(layer)
                .and_then(Option::as_ref)
                .map(Qwen4MoeLayerRuntime::ep_lane_facts),
        )
    }

    /// Device pointers of this rank's compact routed-expert tensors, one per
    /// trunk MoE layer in layer order (what every peer's full table names).
    pub(crate) fn ep_expert_sources(
        &self,
        bundle: &Qwen4Bundle,
    ) -> Result<Vec<Qwen4PeerExpertSource>, Qwen4GpuForwardError> {
        if self.ep.is_none() {
            return Err(invalid(
                "EP expert sources of a forward without an EP placement",
            ));
        }
        let mut sources = Vec::with_capacity(self.moe.len());
        for layer in &bundle.weights.layer_refs {
            if let Some(weights) = layer.ffn.moe() {
                let gate_up = bundle.weights.resident(&weights.experts_gate_up)?;
                let down = bundle.weights.resident(&weights.experts_down)?;
                sources.push(Qwen4PeerExpertSource {
                    gate_up_base: gate_up.buf.as_ptr() as usize,
                    gate_up_bytes: gate_up.buf.size(),
                    down_base: down.buf.as_ptr() as usize,
                    down_bytes: down.buf.size(),
                });
            }
        }
        if sources.len() != self.moe_count() {
            return Err(invalid(format!(
                "EP expert sources cover {} layers, the forward has {}",
                sources.len(),
                self.moe_count()
            )));
        }
        Ok(sources)
    }

    /// The expanded-row gather buffers, indexed by [`qwen4_ep_buffer_index`].
    pub(crate) fn ep_gather_buffers(&self) -> Option<[&GpuTensor; QWEN4_EP_GATHER_BUFFERS]> {
        self.ep
            .as_ref()
            .map(|ep| std::array::from_fn(|index| &ep.buffers[index]))
    }

    /// This rank's EP binding for a head MoE runtime: its placement, the
    /// zero dummy expert and its seal (`None` on the single-device route).
    pub(crate) fn ep_binding(
        &self,
    ) -> Option<(Qwen4EpPlacement, &Qwen4EpDummyExpert, &Qwen4EpRankSeal)> {
        self.ep
            .as_ref()
            .map(|ep| (ep.placement, &ep.dummy, &ep.seal))
    }

    /// What an EP rank's MTP head borrows from this forward at call time: the
    /// row gather and the gather buffers (`None` outside EP or before
    /// `attach_ep`).
    pub(crate) fn mtp_ep_gather(&self) -> Option<MtpEpGather<'_>> {
        let ep = self.ep.as_ref()?;
        Some(MtpEpGather {
            hook: ep.gather.as_deref()?,
            buffers: std::array::from_fn(|index| &ep.buffers[index]),
        })
    }

    /// What the rank's row-gather hook needs to run balanced ownership: `None`
    /// when `HIPFIRE_EP_BALANCED_OWNERS=0` was set at [`Self::attach_ep`]
    /// (default on under EP) or no trunk MoE layer is host-mapped. The
    /// `slot_owner` view is non-owning; this forward frees the buffer.
    pub(crate) fn balanced_plan(&self) -> Option<Qwen4BalancedPlan> {
        let ep = self.ep.as_ref()?;
        let owners = ep.slot_owner.as_ref()?;
        let layers: Vec<(usize, Qwen4EpOwnerTables)> = ep
            .balanced
            .iter()
            .flatten()
            .map(|layer| (layer.label, layer.tables))
            .collect();
        if layers.is_empty() {
            return None;
        }
        Some(Qwen4BalancedPlan {
            layers,
            slot_owner: GpuTensor {
                // SAFETY: a `Borrowed` alias of the buffer `Qwen4ForwardEp`
                // owns until `free_gpu`; the hook that holds it lives inside
                // that forward and is never used after the forward is freed.
                buf: unsafe { owners.buf.alias() },
                shape: owners.shape.clone(),
                dtype: owners.dtype,
            },
        })
    }

    /// After every rank is loaded and P2P is enabled on every rank: build this
    /// rank's full (P2P) chunk tables from all ranks' sources
    /// (`peers[rank][moe_layer]`), take the symmetric-IU4 verdict a single
    /// device takes at construction, and install the row gather the one-row and few-row
    /// forwards use.
    pub(crate) fn attach_ep(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        peers: &[Vec<Qwen4PeerExpertSource>],
        gather: Arc<dyn MoeRowGather + Send + Sync>,
    ) -> Result<(), Qwen4GpuForwardError> {
        if self.share.is_some() {
            return Err(invalid(
                "expert parallelism cannot attach to a decode-share forward (bank-selected scratch)",
            ));
        }
        let layers = self.moe_count();
        let max_chunk = self.prompt_chunk_rows();
        // The EP decode runtime's route-selecting properties per MoE ordinal
        // (host-mapped, symmetric, all-MQ4 gate/up): a balanced layer's live
        // runtime must select the same sealed route.
        let moe_traits: Vec<(bool, bool, bool)> = self
            .moe
            .iter()
            .flatten()
            .map(|moe| (moe.host_mapped, moe.symmetric, moe.experts_all_gate_up_mq4))
            .collect();
        let Some(ep) = self.ep.as_mut() else {
            return Err(invalid("attach_ep on a forward without an EP placement"));
        };
        if !ep.moe_full.is_empty() || ep.gather.is_some() {
            return Err(invalid("EP is already attached"));
        }
        let placement = ep.placement;
        if peers.len() != placement.ranks {
            return Err(invalid(format!(
                "attach_ep got {} peer source sets for {} ranks",
                peers.len(),
                placement.ranks
            )));
        }
        for (rank, sources) in peers.iter().enumerate() {
            if sources.len() != layers {
                return Err(invalid(format!(
                    "attach_ep rank {rank} lists {} MoE layers, expected {layers}",
                    sources.len()
                )));
            }
        }
        let config = &bundle.config;
        let mut balanced_on = qwen4_ep_balanced_owners_enabled();
        if balanced_on {
            if let Some(reason) = qwen4_ep_balanced_owners_refusal(&gpu.arch, placement.ranks) {
                // Nothing is built, allocated or loaded: the static split is exact.
                eprintln!(
                    "qwen4 EP rank {}/{}: balanced ownership not admitted ({reason}); static split",
                    placement.rank, placement.ranks
                );
                balanced_on = false;
            }
        }
        let dummy = ep.dummy.source();
        let seal = ep.seal.clone();
        let layer_peers_of =
            |ordinal: usize| -> Result<Vec<Qwen4PeerExpertSource>, Qwen4GpuForwardError> {
                peers
                    .iter()
                    .map(|sources| {
                        sources.get(ordinal).copied().ok_or_else(|| {
                            invalid(format!("attach_ep is missing MoE layer {ordinal}"))
                        })
                    })
                    .collect()
            };
        let mut full: Vec<Qwen4MoeLayerRuntime> = Vec::with_capacity(layers);
        let mut balanced: Vec<Option<Qwen4BalancedLayer>> = Vec::new();
        let mut slot_owner: Option<GpuTensor> = None;
        let result = (|| -> Result<(), Qwen4GpuForwardError> {
            let mut ordinal = 0usize;
            for layer in &bundle.weights.layer_refs {
                if let Some(weights) = layer.ffn.moe() {
                    let layer_peers = layer_peers_of(ordinal)?;
                    full.push(Qwen4MoeLayerRuntime::from_moe_full(
                        gpu,
                        &bundle.weights,
                        weights,
                        layer.layer,
                        config,
                        placement,
                        &layer_peers,
                        &seal,
                    )?);
                    ordinal += 1;
                }
            }
            // The single-device construction's symmetric-IU4 verdict, taken
            // on the full tables so a chunk forward takes the same route
            // (symmetric IU4 or F16) as one device.
            if !full.is_empty() && gpu.qwen4_moe_sym_iu4_requested() {
                let mut verified = 0usize;
                for layer in &mut full {
                    verified += usize::from(layer.verify_symmetric(gpu, config)?);
                }
                eprintln!(
                    "  qwen4 symmetric IU4 MoE (ep rank {}/{}): {verified}/{} layers verified symmetric",
                    placement.rank,
                    placement.ranks,
                    full.len()
                );
                if verified > 0 {
                    gpu.reserve_qwen4_moe_sym(
                        max_chunk,
                        config.num_experts_per_tok,
                        config.num_experts,
                        config.hidden_size,
                        config.moe_intermediate_size,
                    )?;
                }
            }
            // Balanced ownership (default on under EP; `HIPFIRE_EP_BALANCED_OWNERS=0` disables): every
            // host-mapped trunk layer gets a live runtime of the same full
            // addressing, whose tables the per-call owner assignment
            // rewrites. Nothing is built, allocated or loaded when off.
            if balanced_on {
                let mut ordinal = 0usize;
                for layer in &bundle.weights.layer_refs {
                    let Some(weights) = layer.ffn.moe() else {
                        continue;
                    };
                    let (host_mapped, symmetric, all_mq4) = moe_traits[ordinal];
                    if !host_mapped {
                        balanced.push(None);
                        ordinal += 1;
                        continue;
                    }
                    let owners = match slot_owner.as_ref() {
                        Some(owners) => owners.buf.as_ptr() as usize,
                        None => {
                            let tensor = gpu.zeros(
                                &[QWEN4_EP_GATHER_MAX_ROWS * config.num_experts_per_tok],
                                DType::F32,
                            )?;
                            let address = tensor.buf.as_ptr() as usize;
                            slot_owner = Some(tensor);
                            address
                        }
                    };
                    let live = Qwen4MoeLayerRuntime::from_moe_full(
                        gpu,
                        &bundle.weights,
                        weights,
                        layer.layer,
                        config,
                        placement,
                        &layer_peers_of(ordinal)?,
                        &seal,
                    )?;
                    if live.symmetric != symmetric
                        || live.host_mapped != host_mapped
                        || live.experts_all_gate_up_mq4 != all_mq4
                    {
                        let _ = live.free_gpu(gpu);
                        return Err(invalid(format!(
                            "layer {} balanced runtime selects a different sealed route than the EP decode runtime",
                            layer.layer
                        )));
                    }
                    let tables = Qwen4EpOwnerTables {
                        full_gate_up: full[ordinal].expert_gate_up_ptrs.buf.as_ptr() as usize,
                        full_down: full[ordinal].expert_down_ptrs.buf.as_ptr() as usize,
                        live_gate_up: live.expert_gate_up_ptrs.buf.as_ptr() as usize,
                        live_down: live.expert_down_ptrs.buf.as_ptr() as usize,
                        slot_owner: owners,
                        dummy_gate_up: dummy.gate_up_base,
                        dummy_down: dummy.down_base,
                    };
                    balanced.push(Some(Qwen4BalancedLayer {
                        live,
                        label: layer.layer,
                        tables,
                    }));
                    ordinal += 1;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            for layer in full {
                let _ = layer.free_gpu(gpu);
            }
            for layer in balanced.into_iter().flatten() {
                let _ = layer.live.free_gpu(gpu);
            }
            if let Some(tensor) = slot_owner {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        if balanced_on {
            eprintln!(
                "qwen4 EP rank {}/{}: balanced ownership on {} host-mapped MoE layers",
                placement.rank,
                placement.ranks,
                balanced.iter().flatten().count()
            );
        }
        if let Some(route) = ep.tile.as_mut() {
            // The compact table carries no verdict of its own under EP (the verification
            // above ran on the full tables; same experts, same bytes): the mesh compares
            // all ranks' vectors before anything is propagated to the compact layers. The
            // F16 WMMA route (gfx1100) reads no symmetric header: its verdict is the
            // layer's expert formats.
            let wmma = route.wmma;
            route.verdicts = full
                .iter()
                .map(|layer| {
                    if wmma {
                        layer.f16_wmma_tile_eligible()
                    } else {
                        layer.symmetric
                    }
                })
                .collect();
        }
        ep.moe_full = full;
        ep.balanced = balanced;
        ep.slot_owner = slot_owner;
        ep.gather = Some(gather);
        Ok(())
    }

    /// This rank's grouped-row buffer (the peer reads it) and the tile capacity the
    /// exchange is registered for, or `None` when not admitted.
    pub(crate) fn ep_tile_registration(&self) -> Option<(&GpuTensor, usize)> {
        let route = self.ep.as_ref()?.tile.as_ref()?;
        let root = self
            .ep_reserve
            .as_ref()
            .map_or(&self.scratch.moe_y_down_grouped, |reserve| {
                &reserve.exchange_root
            });
        Some((root, route.max_tiles))
    }

    /// The per-layer symmetric verdicts taken in `attach_ep` (`None` when not admitted).
    pub(crate) fn ep_tile_verdicts(&self) -> Option<Vec<bool>> {
        Some(self.ep.as_ref()?.tile.as_ref()?.verdicts.clone())
    }

    /// A non-owning view of the rank's owner table for the transport hook (owned by the
    /// forward, freed with it; the hook is never used after the forward is freed).
    pub(crate) fn ep_tile_owner_table_view(&self) -> Option<GpuTensor> {
        let table = &self.ep.as_ref()?.tile.as_ref()?.owner_table;
        Some(GpuTensor {
            // SAFETY: a `Borrowed` alias of the buffer `Qwen4TileRoute` owns until
            // `free_gpu`; the hook that holds it lives inside this forward.
            buf: unsafe { table.buf.alias() },
            shape: table.shape.clone(),
            dtype: table.dtype,
        })
    }

    /// Install the agreed transport. On gfx1201 every layer's compact runtime takes the
    /// symmetric verdict (same experts, same bytes as the full tables it was taken on), so a
    /// chunk of at least 512 rows binds the compact table with the tile exchange; on gfx1100
    /// the compact layers keep their plain grouped declaration (the WMMA arms are chosen from
    /// the device and the row count). Refused unless every layer's verdict is true (the mesh
    /// checked every rank agrees).
    pub(crate) fn install_tile_exchange(
        &mut self,
        hook: Arc<dyn MoeTileExchange + Send + Sync>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let layers = self.moe_count();
        let Some(route) = self.ep.as_mut().and_then(|ep| ep.tile.as_mut()) else {
            return Err(invalid(
                "owner-routed prefill was not admitted on this rank",
            ));
        };
        if route.verdicts.len() != layers || !route.verdicts.iter().all(|&admitted| admitted) {
            return Err(invalid(
                "owner-routed prefill needs every layer admitted on the full tables \
                 (symmetric IU4 headers, or QT44/QT53 experts for the F16 WMMA route)",
            ));
        }
        let wmma = route.wmma;
        route.hook = Some(hook);
        if !wmma {
            for layer in self
                .moe
                .iter_mut()
                .flatten()
                .chain(self.moe_borrow.iter_mut().flatten().flatten())
            {
                layer.symmetric = true;
            }
        }
        Ok(())
    }

    /// Name the tokens that follow the next forward (at most one chunk is
    /// kept); see the `ple_lookahead` field.
    pub(crate) fn set_ple_lookahead(&mut self, tokens: &[u32]) {
        let rows = self.prompt_chunk_rows();
        self.ple_lookahead.clear();
        self.ple_lookahead
            .extend_from_slice(&tokens[..tokens.len().min(rows)]);
    }

    /// Whether this forward runs decode-share banks (a resident bank plus
    /// borrowed prefill banks).
    pub(crate) fn decode_share_active(&self) -> bool {
        self.share.is_some()
    }

    /// Rows of one prompt tile: the prompt schedule capacity every tiling loop,
    /// chunk-size query and session boundary stride uses. On a decode-share
    /// load the resident bank holds only
    /// [`QWEN4_DECODE_SHARE_RESIDENT_ROWS`] rows while a prompt tile runs on a
    /// borrowed prefill bank, so capacity is never the selected bank's
    /// `max_chunk`; everywhere else it is `scratch.max_chunk`.
    pub(crate) fn prompt_chunk_rows(&self) -> usize {
        if let Some(reserve) = self.ep_reserve.as_ref() {
            return reserve.banks.prompt_rows();
        }
        prompt_chunk_rows_of(self.share.as_ref(), &self.scratch)
    }

    /// Whether this call continues a dirty borrow epoch inside a whole-prompt
    /// entry that itself borrowed: once a tile of the entry has dirtied the
    /// donor tail, every later tile of the same entry (a tail below the
    /// resident rows included) stays on a borrowed bank and the BorrowPrefill
    /// runtimes, so the restore happens once, at the next resident call after
    /// the prompts, never mid-prompt (design 5.2). A dirty loan left by an
    /// earlier, finished entry does not continue (`prompt_borrowed` is clear):
    /// the first sub-resident call of a later entry (an MTP verify block, a
    /// draft forward, a one-row capture) restores first, as AR decode does.
    /// See [`decode_share_continues_epoch`].
    fn share_continues_epoch(&self) -> bool {
        self.share.as_ref().is_some_and(|share| {
            decode_share_continues_epoch(
                share.prompt_depth,
                share.prompt_borrowed,
                share
                    .loan
                    .as_ref()
                    .is_some_and(|loan| loan.controller.dirty_extent().is_some()),
            )
        })
    }

    /// The prefill class a tile of `rows` rows borrows; `None` runs the
    /// resident bank (every tile below the resident rows outside a dirty
    /// prompt epoch, and any call on a load without decode-share). A small
    /// continuation tile of a dirty epoch takes the smallest class.
    fn share_class_for_tile(&self, rows: usize) -> Result<Option<usize>, Qwen4GpuForwardError> {
        let Some(share) = self.share.as_ref() else {
            return Ok(None);
        };
        if rows < QWEN4_DECODE_SHARE_RESIDENT_ROWS {
            return Ok(self.share_continues_epoch().then_some(0));
        }
        decode_share_class_for_rows(&share.classes, rows)
            .map(Some)
            .ok_or_else(|| {
                invalid(format!(
                    "a {rows}-row tile exceeds the largest decode-share prefill bank ({} rows)",
                    share.classes.last().copied().unwrap_or(0)
                ))
            })
    }

    /// Whether the call is a decode-share prompt tile: the single predicate
    /// behind both scratch-bank selection (`share` banks) and BorrowPrefill
    /// runtime selection (`moe_borrow`). A tile of at least the resident rows
    /// ([`decode_share_prefill_tile`]) always borrows; a smaller non-argmax,
    /// non-lane, non-mesh call borrows only as the tail of an entry that
    /// itself borrowed ([`Self::share_continues_epoch`]). See
    /// [`decode_share_borrows_tile`].
    fn share_prefill_tile(&self, tokens: &[u32], argmax_of: Option<&GpuTensor>) -> bool {
        if self.share.is_none() && self.moe_borrow.is_empty() {
            return false;
        }
        decode_share_borrows_tile(
            tokens.len(),
            argmax_of.is_some(),
            self.lane_scope,
            self.mesh_route_rows.is_some(),
            self.share_continues_epoch(),
        )
    }

    /// The BorrowPrefill class view the prompt tile about to run reads
    /// ([`borrow_view_class`]): after `share_acquire`, the loan's dirty extent
    /// holds the largest class span of the epoch, this tile included. `bind`
    /// (no loan) reads the tile's own class view. `tile_class` is the bank
    /// class of a decode-share load, `None` for `bind`.
    fn borrow_view_for_tile(&self, tile_class: Option<usize>, rows: usize) -> usize {
        let views = self.moe_borrow.len();
        let tile_class = tile_class
            .or_else(|| decode_share_class_for_rows(&self.borrow_class_rows, rows))
            .unwrap_or(views.saturating_sub(1));
        let (spans, extent) = match self.share.as_ref() {
            Some(share) => (
                share.class_spans.as_slice(),
                share
                    .loan
                    .as_ref()
                    .and_then(|loan| loan.controller.dirty_extent()),
            ),
            None => (&[][..], None),
        };
        borrow_view_class(views, tile_class, spans, extent)
    }

    /// The per-layer MoE runtimes the running tile reads: the BorrowPrefill
    /// twins during a decode-share prompt tile, the Resident runtimes in every
    /// other forward (decode, verify, lanes, EP, mesh, small tiles, attach).
    pub(crate) fn tile_moe(&self) -> &[Option<Qwen4MoeLayerRuntime>] {
        tile_moe_of(
            self.borrow_runtimes_active,
            self.borrow_view,
            &self.moe,
            &self.moe_borrow,
        )
    }

    /// Every class set of `moe_borrow` must mirror `moe` layer for layer and
    /// report the same `host_mapped` bit (verdict change 3: layers without a
    /// host expert never donate, so the layer bit that selects the stage and
    /// the host wait is view independent). A lease with several class views
    /// must carry the spans the forward's banks dirty (`share_spans`, absent for
    /// `bind`). Logs the BorrowPrefill runtime census and the donors each class
    /// dirties.
    fn check_borrow_runtimes(
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        borrow: &[Vec<Option<Qwen4MoeLayerRuntime>>],
        share_spans: Option<&[u64]>,
    ) -> Result<(), Qwen4GpuForwardError> {
        use crate::expert_tier_owners::ExpertReadView;
        use crate::expert_tier_plan::Placement;
        let lease = bundle
            .weights
            .expert_tiers()
            .ok_or_else(|| invalid("decode-share runtimes without a tier lease"))?;
        if borrow.len() != lease.borrow_classes() {
            return Err(invalid(format!(
                "decode-share: {} BorrowPrefill runtime sets for {} lease class views",
                borrow.len(),
                lease.borrow_classes()
            )));
        }
        if let Some(spans) = share_spans {
            if borrow.len() > 1 && lease.borrow_class_spans() != spans {
                return Err(invalid(format!(
                    "decode-share: the lease's class spans {:?} differ from the scratch banks' {spans:?}",
                    lease.borrow_class_spans()
                )));
            }
        }
        let (mut layers, mut donor_layers) = (0usize, 0usize);
        for (class, runtimes) in borrow.iter().enumerate() {
            let last = class + 1 == borrow.len();
            for (layer, (resident, borrowed)) in moe.iter().zip(runtimes).enumerate() {
                let (resident, borrowed) = match (resident, borrowed) {
                    (Some(resident), Some(borrowed)) => (resident, borrowed),
                    (None, None) => continue,
                    _ => {
                        return Err(invalid(format!(
                            "decode-share: layer {layer} has a runtime in one read view only"
                        )))
                    }
                };
                if resident.host_mapped() != borrowed.host_mapped() {
                    return Err(invalid(format!(
                        "decode-share: layer {layer} host_mapped is {} in the Resident view and {} in \
                         the BorrowPrefill class {class} view (a layer without a host expert must not donate)",
                        resident.host_mapped(),
                        borrowed.host_mapped()
                    )));
                }
                if !last {
                    continue;
                }
                layers += 1;
                let (Some(res_view), Some(bor_view)) = (
                    lease.layer_view(layer, ExpertReadView::Resident),
                    lease.layer_view(layer, ExpertReadView::BorrowPrefill(class)),
                ) else {
                    return Err(invalid(format!(
                        "decode-share: layer {layer} lacks a tier view"
                    )));
                };
                if (0..bundle.config.num_experts).any(|eid| {
                    res_view.placement(eid) == Some(Placement::Device)
                        && bor_view.placement(eid) == Some(Placement::HostMapped)
                }) {
                    donor_layers += 1;
                }
            }
        }
        eprintln!(
            "  qwen4 decode-share: BorrowPrefill runtimes for {layers} tier layers (donors in {donor_layers} layers), {} class views",
            borrow.len()
        );
        Ok(())
    }

    /// Before a prompt tile swaps prefill class `class` in: the loan
    /// controller marks the donor tail dirty (and zero-fills the class's span
    /// growth) before the tile's first write. Refuses before any write under
    /// a stream capture or a poisoned loan. Nothing over an owned backing.
    fn share_acquire(&mut self, gpu: &mut Gpu, class: usize) -> Result<(), Qwen4GpuForwardError> {
        let Some(share) = self.share.as_mut() else {
            return Ok(());
        };
        let span = share.class_spans[class];
        match share.loan.as_mut() {
            Some(loan) => loan
                .with_ops(gpu, |controller, ops| controller.acquire(ops, span))
                .map_err(invalid),
            None => Ok(()),
        }
    }

    /// Before a call that runs on the resident bank and the Resident runtimes
    /// (decode, verify, lanes, every tile below the resident rows outside a
    /// prompt): the donor tail must be whole, so a dirty tail (a prompt that
    /// ended `Ok` leaves it dirty) is restored, whole dirtied prefix enqueued
    /// and host-waited, and a pending restore is waited. This is where the
    /// second-token latency after a long prompt pays the restore. Runs before the retained-eligibility
    /// predicate and any replay submit.
    pub(crate) fn share_require_resident(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<(), Qwen4GpuForwardError> {
        if let Some(share) = self.share.as_mut() {
            // The donor tail is about to be whole: the epoch is clean.
            share.prompt_borrowed = false;
        }
        match self.share.as_mut().and_then(|share| share.loan.as_mut()) {
            Some(loan) => loan
                .with_ops(gpu, |controller, ops| controller.require_resident(ops))
                .map_err(invalid),
            None => Ok(()),
        }
    }

    /// After every tile, with the resident bank back in `scratch` and whether
    /// the tile succeeded: a failed tile must not leave dirty weights behind
    /// even if the caller abandons the prompt (blocking recovery). `Err` only
    /// when that recovery failed (the loan is poisoned).
    fn share_after_tile(&mut self, gpu: &mut Gpu, ok: bool) -> Result<(), Qwen4GpuForwardError> {
        match self.share.as_mut().and_then(|share| share.loan.as_mut()) {
            Some(loan) => loan
                .with_ops(gpu, |controller, ops| controller.after_tile(ops, ok))
                .map_err(invalid),
            None => Ok(()),
        }
    }

    /// `Err` naming the reason when the donor loan is poisoned: every forward,
    /// on any route, refuses.
    fn share_check(&self) -> Result<(), Qwen4GpuForwardError> {
        match self.share.as_ref().and_then(|share| share.loan.as_ref()) {
            Some(loan) => loan.controller.check().map_err(invalid),
            None => Ok(()),
        }
    }

    /// Whether the donor tail is whole (always true without a donor loan).
    fn share_state_resident(&self) -> bool {
        self.share
            .as_ref()
            .and_then(|share| share.loan.as_ref())
            .map_or(true, |loan| loan.controller.is_resident())
    }

    /// A whole-prompt entry begins: nested entries (a driver that issues
    /// several `forward_chunk` calls inside its own whole-prompt loop) count,
    /// and only the outermost one's [`Self::share_prompt_end`] finishes the
    /// prompt. No-op without decode-share.
    pub(crate) fn share_prompt_begin(&mut self) {
        self.ep_reserve_prompt_begin();
        if let Some(share) = self.share.as_mut() {
            share.prompt_depth += 1;
        }
    }

    /// [`Self::share_prompt_begin`] for the native-MTP fill: an outermost entry
    /// it opens is recorded with receipt phase `mtp_fill` (a nested one keeps
    /// the phase of its outermost entry).
    pub(crate) fn share_prompt_begin_mtp_fill(&mut self) {
        self.receipt_mtp_fill = true;
        self.share_prompt_begin();
        self.receipt_mtp_fill = false;
    }

    /// A whole-prompt entry ends (`ok`: it returned `Ok`); the outermost one
    /// runs [`Self::share_finish_prompt`]. No-op without decode-share.
    pub(crate) fn share_prompt_end(
        &mut self,
        gpu: &mut Gpu,
        ok: bool,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.ep_ordinary_prompt_end(ok);
        if self.ep_reserve.is_some() {
            return self.ep_reserve_prompt_end(gpu, ok);
        }
        let Some(share) = self.share.as_mut() else {
            return Ok(());
        };
        share.prompt_depth = share.prompt_depth.saturating_sub(1);
        if share.prompt_depth == 0 {
            // The entry that borrowed is over, whatever `finish_prompt` returns:
            // a dirty tail it leaves is no later entry's continuation.
            share.prompt_borrowed = false;
            self.share_finish_prompt(gpu, ok)
        } else {
            Ok(())
        }
    }

    /// The last scratch consumer of a whole prompt has run. A dirty tail of a
    /// prompt that ended `Ok` stays dirty with nothing enqueued (the first
    /// token came from the prefill head; a following prompt's tiles of at
    /// least the resident rows continue the same borrow, any other call of a
    /// later entry and the next [`Self::share_require_resident`] restore it;
    /// `HIPFIRE_QWEN4_DECODE_SHARE_SYNC=1` enqueues and waits here). A prompt
    /// that ended with an error takes the blocking recovery.
    pub(crate) fn share_finish_prompt(
        &mut self,
        gpu: &mut Gpu,
        ok: bool,
    ) -> Result<(), Qwen4GpuForwardError> {
        match self.share.as_mut().and_then(|share| share.loan.as_mut()) {
            Some(loan) => loan
                .with_ops(gpu, |controller, ops| controller.finish_prompt(ops, ok))
                .map_err(invalid),
            None => Ok(()),
        }
    }

    /// Device bytes this forward's scratch is charged: the resident bank, the
    /// persistent token ids and the speculative buffers on a decode-share load
    /// ([`qwen4_decode_share_dedicated_bytes`]); the whole forward otherwise.
    pub(crate) fn forward_device_bytes(&self, config: &Qwen4Config) -> Option<u64> {
        if let Some(reserve) = self.ep_reserve.as_ref() {
            return reserve.plan.forward.dedicated_bytes().ok();
        }
        match self.share.as_ref() {
            Some(share) => {
                qwen4_decode_share_dedicated_bytes(config, share.max_chunk, self.scratch.mode)
            }
            None => qwen4_forward_device_bytes(config, self.scratch.max_chunk, self.scratch.mode),
        }
    }

    /// Hand the forward the S1 slow-link profile the load admitted
    /// (`Qwen4Bundle::attach_slowlink_profile`, before attach). The profile is
    /// immutable for the forward's life; nothing is allocated here.
    pub(crate) fn set_slowlink_profile(
        &mut self,
        profile: crate::slowlink_admission::SlowlinkProfile,
    ) {
        self.slowlink.profile = Some(profile);
    }

    /// The admitted S1 profile, `None` on every load that did not take S1.
    pub(crate) fn slowlink_profile(&self) -> Option<&crate::slowlink_admission::SlowlinkProfile> {
        self.slowlink.profile.as_ref()
    }

    /// The layers this forward runs (the whole trunk for a single-band forward).
    pub(crate) fn band_layers(&self) -> std::ops::Range<usize> {
        self.scope.band.clone()
    }

    /// Why this forward refuses an S1 prompt: a mode whose state or weights S1
    /// does not own. Host only, no mutation.
    pub(crate) fn slowlink_forward_blocker(&self, config: &Qwen4Config) -> Option<&'static str> {
        if self.ep.is_some() {
            Some("expert parallelism")
        } else if self.qsa_tap.is_some() {
            Some("the QSA parity tap")
        } else if self.route_trace.is_some() {
            Some("the routing trace")
        } else if self.cut_layers.is_some() {
            Some(CUT_LAYERS_ENV)
        } else if self.lane_scope {
            Some("lane scope")
        } else if self.mesh_route_rows.is_some() {
            Some("a mesh forward")
        } else if self.npu_trunk_loaded() {
            Some("the NPU trunk")
        } else if self.scope != Qwen4BandScope::single(config) {
            Some("a pipeline band")
        } else {
            None
        }
    }

    /// First effects of an S1 prompt, before it touches any state: restore a
    /// donor a failed staged forward left overwritten (resident layers run
    /// here), retire the early token-id copy (`token_ids` is rewritten per
    /// tile) and drop the stale retained PLE ticket and lookahead (S1 fetches
    /// every tile's rows itself, from its private history cursor). Returns the
    /// retained ticket for the first tile to adopt.
    pub(crate) fn slowlink_begin(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<Option<RowTicket>, Qwen4GpuForwardError> {
        if self.share.is_some() {
            return Err(invalid(
                "S1 slow-link prefill cannot run on a decode-share forward (bank-selected scratch)",
            ));
        }
        if let Some(stage) = self.expert_stage.as_ref() {
            stage.restore_blocking(gpu)?;
        }
        if let Some(readback) = self.token_readback.as_mut() {
            readback.complete(&gpu.hip)?;
        }
        self.ple_lookahead.clear();
        self.uploaded_id_rows = 0;
        Ok(self.ple_ahead.take())
    }

    /// A device-side effect of S1 rewrote `token_ids`: no claim on the ids an
    /// earlier forward uploaded.
    pub(crate) fn slowlink_note_ids_rewritten(&mut self) {
        self.uploaded_id_rows = 0;
    }

    /// Device bytes of the forward's own mixer-tile scratch (every owned slot
    /// of the plan): the S1 receipt's `reserve.tile`.
    pub(crate) fn slowlink_tile_scratch_bytes(&self) -> u64 {
        self.scratch
            .slots
            .iter()
            .filter(|slot| slot.owner.is_none())
            .filter_map(|slot| slot.bytes().ok())
            .map(|bytes| bytes as u64)
            .sum()
    }

    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
    /// One named production-primitive check on this forward's admitted S1
    /// transport and dispatch owners (`Qwen4Bundle::slowlink_primitive_selftest`
    /// is the public facade). Returns exactly
    /// `{"name","status":"pass"|"fail","mismatches","detail"}` from measured
    /// device results; `Err` for a case that cannot run, never a default pass.
    pub(crate) fn slowlink_primitive_selftest(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<serde_json::Value, Qwen4GpuForwardError> {
        crate::slowlink_prefill::primitive_selftest(self, bundle, gpu, name)
    }

    /// Teardown only: free the forward right before the bundle weights it
    /// ran on (`Qwen4Bundle::free_gpu`). After a failed staged forward
    /// whose recovery also failed, two resident expert layers may still hold
    /// another layer's bytes, and nothing here restores them.
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        // Tier-split layers own pinned transfer storage and enqueued slot
        // uploads: the device is drained once, before any layer releases its
        // executor (the layers drain again before freeing their buffers).
        let cpu_split_layers = self.cpu_split_layers;
        let Qwen4GpuForward {
            mut scratch,
            share,
            ep_reserve,
            host_ple,
            mesh_tokens,
            lane_tokens,
            moe,
            moe_borrow,
            expert_stage,
            ep,
            token_readback,
            decode_q8,
            route_trace,
            trunk_zba,
            slowlink,
            #[cfg(target_os = "linux")]
            npu_trunk,
            ..
        } = self;
        // Drain both streams before releasing stage tables, events or scratch.
        // A donor a failed forward left overwritten is not restored: the
        // bundle attaches its forward once and frees it only right before
        // the weights (`Qwen4Bundle::free_gpu`), so nothing reads it again.
        if expert_stage.is_some() || ep.is_some() || cpu_split_layers > 0 {
            gpu.hip.device_synchronize()?;
        }
        // The S1 expert page ring and prompt storage drain their own streams
        // before freeing; a failed teardown is reported, never silenced.
        let slowlink_error = slowlink.free_gpu(gpu).err();
        // The PLE staging bytes may still be read by an enqueued upload:
        // retire it before the device buffers and the host bytes go.
        let mut first = host_ple.free_gpu(gpu);
        if let Some(error) = slowlink_error {
            first.get_or_insert(hip_bridge::HipError::new(
                999,
                &format!("slow-link expert ring teardown failed: {error}"),
            ));
        }
        if let Some(error) = mesh_tokens.free_gpu(gpu) {
            first.get_or_insert(error);
        }
        if let Some(error) = lane_tokens.free_gpu(gpu) {
            first.get_or_insert(error);
        }
        #[cfg(target_os = "linux")]
        if let Some(npu) = npu_trunk {
            if let Err(e) = npu.free(gpu) {
                first.get_or_insert(e);
            }
        }
        let mut share = share;
        if let Some(share) = share.as_mut() {
            // Never free through a prefill bank: the resident bank is back in
            // `scratch` (and its owners are freed once, below).
            share.swap_out(&mut scratch);
            eprintln!(
                "  qwen4 decode-share: {} prompt tiles ran on borrowed prefill banks ({} bank swaps)",
                share.stats.tiles_borrowed, share.stats.bank_swaps
            );
            // The donors must be whole before the lease can go. A restore that
            // cannot be proven complete leaks the loan (claim, lease retention,
            // completion event) instead of freeing under a pending DMA.
            if let Some(loan) = share.loan.as_mut() {
                let torn_down = loan.with_ops(gpu, |controller, ops| controller.teardown(ops));
                eprintln!(
                    "  qwen4 decode-share: donor loan {}",
                    loan.controller.summary()
                );
                if let Err(error) = torn_down {
                    eprintln!(
                        "  qwen4 decode-share: teardown restore not proven ({error}); leaking the \
                         donor loan rather than freeing under a possibly pending copy"
                    );
                    std::mem::forget(share.loan.take());
                }
            }
        }
        // EP reserve: the swapped-in bank is parked back and every donor restored (the loan's
        // synchronous teardown) before the scratch is freed. A restore that cannot be proven
        // complete leaks the loan instead of freeing it under a possibly pending copy.
        let mut reserve_teardown = None;
        if let Some(mut reserve) = ep_reserve {
            if let Err(error) = reserve.banks.swap_out(&mut scratch) {
                first.get_or_insert(hip_bridge::HipError::new(999, &error.to_string()));
            }
            reserve_teardown = Some(reserve.teardown(gpu));
        }
        if let Some(error) = scratch.free_gpu(gpu) {
            first.get_or_insert(error);
        }
        let mut reserve_loan = None;
        if let Some(ep_reserve_forward::EpReserveTeardown { banks, loan }) = reserve_teardown {
            if let Some(error) = banks.release(gpu) {
                first.get_or_insert(error);
            }
            reserve_loan = loan;
        }
        if let Some(error) = share.and_then(|share| share.free_gpu(gpu)) {
            first.get_or_insert(error);
        }
        if let Some(stage) = expert_stage {
            if let Some(error) = stage.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some((loan, permit)) = reserve_loan {
            if let Some(error) = loan.release(gpu, permit) {
                first.get_or_insert(error);
            }
        }
        if let Some(trace) = route_trace {
            if let Err(error) = gpu.free_tensor(trace.device) {
                first.get_or_insert(error);
            }
        }
        gpu.qwen4_trunk_iu4 = false;
        for tensor in decode_q8.into_iter().chain(trunk_zba.into_iter().flatten()) {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        if let Some(mut readback) = token_readback {
            // Drain the id copy first: the pinned buffer drops with `readback`.
            let drained = match readback.pinned.as_mut() {
                Some(pinned) => pinned.complete(&gpu.hip),
                None => Ok(()),
            };
            for result in [
                drained,
                gpu.hip.stream_synchronize(&readback.stream),
                gpu.hip.stream_destroy(readback.stream),
                gpu.hip.event_destroy(readback.event),
            ] {
                if let Err(error) = result {
                    first.get_or_insert(error);
                }
            }
        }
        for layer in moe.into_iter().flatten() {
            if let Some(error) = layer.free_gpu(gpu) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        for layer in moe_borrow.into_iter().flatten().flatten() {
            if let Some(error) = layer.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        if let Some(ep) = ep {
            if let Some(error) = ep.free_gpu(gpu) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Append the traced forward's routed expert ids (see [`RouteTrace`]).
    /// Synchronizes the device: tracing is a measurement mode.
    fn write_route_trace(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        use std::io::Write;
        let Some(trace) = self.route_trace.as_mut() else {
            return Ok(());
        };
        gpu.hip.device_synchronize()?;
        gpu.hip.memcpy_dtoh(&mut trace.host, &trace.device.buf)?;
        let mut line = position.to_string();
        for id in trace.host.chunks_exact(4) {
            line.push(' ');
            line.push_str(&i32::from_ne_bytes([id[0], id[1], id[2], id[3]]).to_string());
        }
        line.push('\n');
        trace
            .out
            .write_all(line.as_bytes())
            .and_then(|_| trace.out.flush())
            .map_err(|error| invalid(format!("{ROUTE_TRACE_ENV}: {error}")))
    }

    /// Token-only request checks shared by every forward entry: non-empty,
    /// within context capacity, inside the embedding vocabulary.
    fn validate_tokens(
        &self,
        bundle: &Qwen4Bundle,
        tokens: &[u32],
    ) -> Result<(), Qwen4GpuForwardError> {
        if tokens.is_empty() {
            return Err(invalid("Qwen4 forward cannot process an empty token slice"));
        }
        let config = &bundle.config;
        let end_position = bundle
            .state
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("Qwen4 forward position overflows"))?;
        if end_position > bundle.state.max_seq_len {
            return Err(invalid(format!(
                "Qwen4 forward end position {end_position} exceeds context capacity {}",
                bundle.state.max_seq_len
            )));
        }
        if tokens
            .iter()
            .copied()
            .any(|token| (token as usize) >= config.vocab_size)
        {
            return Err(invalid(
                "Qwen4 token id is outside the embedding vocabulary",
            ));
        }
        Ok(())
    }

    fn validate_request(
        &self,
        bundle: &Qwen4Bundle,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_rows: Qwen4OutputRows,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.validate_tokens(bundle, tokens)?;
        let config = &bundle.config;
        let requested_rows = output_rows.count(tokens.len());
        let expected_logits = requested_rows
            .checked_mul(config.vocab_size)
            .ok_or_else(|| invalid("Qwen4 logits shape overflows"))?;
        if logits.dtype != DType::F32 || logits.numel() != expected_logits {
            return Err(invalid(format!(
                "logits must be F32 with {expected_logits} elements for {output_rows:?} output"
            )));
        }
        if let Some(top1) = top1 {
            let expected = requested_rows
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or_else(|| invalid("Qwen4 top1 shape overflows"))?;
            if top1.dtype != DType::Raw || top1.buf.size() < expected {
                return Err(invalid(
                    "top1 output must be Raw with one i32 per requested output row",
                ));
            }
        }
        if let Some(capture) = wide_hidden_capture {
            let expected = tokens
                .len()
                .checked_mul(program_dims(config).wide())
                .ok_or_else(|| invalid("Qwen4 wide hidden capture shape overflows"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        Ok(())
    }

    /// The bank a tile of `rows` rows runs on, without swapping it in: the
    /// prefill class holding it on a decode-share load (while no bank is
    /// swapped in), else `scratch`.
    fn scratch_for_tile_rows(&self, rows: usize) -> &Qwen4GpuForwardScratch {
        // EP reserve: a wide tile runs on the parked borrowed bank.
        if let Some(bank) = self
            .ep_reserve
            .as_ref()
            .and_then(|reserve| reserve.banks.bank_for_rows(rows))
        {
            return bank;
        }
        self.share
            .as_ref()
            .filter(|share| share.active.is_none())
            .and_then(|share| {
                decode_share_class_for_rows(&share.classes, rows)
                    .and_then(|class| share.banks[class].as_ref())
            })
            .unwrap_or(&self.scratch)
    }

    fn preflight_output_resources(
        &self,
        bundle: &Qwen4Bundle,
        rows: usize,
        logits: &GpuTensor,
        requested_rows: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let config = &bundle.config;
        let dims = program_dims(config);
        let final_hyper = hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)?;
        let lm_head = dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?;
        // The bank the tile of `rows` rows will run on (the resident bank holds
        // too few rows for a wide tile of a decode-share load).
        let scratch = self.scratch_for_tile_rows(rows);
        let preflight_scratch = layer_scratch(scratch, &scratch.router_logits);
        validate_final_hyper(
            dims,
            &final_hyper,
            &scratch.streams,
            &preflight_scratch,
            rows,
        )
        .map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 final hyper: {error:?}"))
        })?;
        let hidden_elements = rows
            .checked_mul(config.hidden_size)
            .ok_or_else(|| invalid("Qwen4 final hidden shape overflows"))?;
        let final_hidden = view(&scratch.hc_mixed, 0, hidden_elements);
        validate_lm_head(&lm_head, &final_hidden, logits, rows, requested_rows).map_err(
            |error| Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 LM head: {error:?}")),
        )?;
        Ok(())
    }

    pub fn forward_token(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.forward_chunk_inner(
            bundle,
            gpu,
            std::slice::from_ref(&token),
            None,
            Some(logits),
            top1,
            None,
            Qwen4OutputPolicy::Rows(Qwen4OutputRows::Final),
        )
    }

    /// Copy the single wide HC stream row the last ordinary one-token
    /// [`Self::forward_token`] left in `scratch.streams` into `destination`
    /// as F32, with the conversion `forward_chunk` applies for a wide hidden
    /// capture (BF16 stream bits widen, F32 streams copy). The final hyper
    /// reads `streams` into other scratch, so the row is the trunk's output on
    /// the HIP and the retained-replay route alike. `destination` must be F32
    /// with exactly `hc_count * hidden_size` elements. Read by the AR floor's
    /// calibration token and by every retired AR token (whose row the MTP
    /// head appends later); the row is valid until the next forward writes
    /// `streams`.
    pub(crate) fn copy_last_wide_hidden_to(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        destination: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        let wide = program_dims(config).wide();
        if destination.dtype != DType::F32 || destination.numel() != wide {
            return Err(invalid(format!(
                "wide hidden destination must be F32 with exactly {wide} elements"
            )));
        }
        if self.scratch.streams.numel() < wide {
            return Err(invalid(
                "Qwen4 stream scratch is narrower than one wide row",
            ));
        }
        let source = view(&self.scratch.streams, 0, wide);
        let destination = f32_view(destination, 0, wide);
        if gpu.qwen4_bf16_streams(1) {
            rdna_compute::tensor_ops::hc_state_bf16_to_f32(gpu, &source, &destination, wide)?;
        } else {
            gpu.copy_d2d(&source, &destination, source.byte_size())?;
        }
        Ok(())
    }

    /// [`Self::forward_token`] of `token`, or (`None`) of the GPU argmax of
    /// `logits` as the previous forward left them; returns the token. The
    /// argmax is taken only after this forward's step program is built, and
    /// on the HIP body it stays on the device until the PLE layer, so neither
    /// the host build nor the readback leaves the GPU idle.
    pub fn forward_token_or_argmax(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, Qwen4GpuForwardError> {
        let placeholder = token.unwrap_or(0);
        self.forward_chunk_inner(
            bundle,
            gpu,
            std::slice::from_ref(&placeholder),
            token.is_none().then_some(logits),
            Some(logits),
            None,
            None,
            Qwen4OutputPolicy::Rows(Qwen4OutputRows::Final),
        )?;
        // The forward committed its token as the newest PLE history slot.
        Ok(bundle.state.ple_history.previous()[1])
    }

    /// Run bounded forward tiles, writing all logits rows or only the last.
    /// Final-only intermediate tiles still commit model state but skip the head.
    pub fn forward_chunk(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_rows: Qwen4OutputRows,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.share_prompt_begin();
        let result = self.forward_chunk_tiles(
            bundle,
            gpu,
            tokens,
            logits,
            top1,
            wide_hidden_capture,
            output_rows,
        );
        let finish = self.share_prompt_end(gpu, result.is_ok());
        merge_share_result(result, finish)
    }

    /// The tile loop of [`Self::forward_chunk`].
    fn forward_chunk_tiles(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_rows: Qwen4OutputRows,
    ) -> Result<(), Qwen4GpuForwardError> {
        // The caller's lookahead names what follows this whole request, so it
        // belongs to the final tile; earlier tiles look ahead to the next tile.
        let mut caller_lookahead = Some(std::mem::take(&mut self.ple_lookahead));
        self.validate_request(
            bundle,
            tokens,
            logits,
            top1,
            wide_hidden_capture,
            output_rows,
        )?;
        self.plan_npu_trunk_request(gpu, tokens.len(), bundle.state.position)?;
        // Start the first tile's PLE rows now, ahead of the output-resource
        // preflight.
        self.prefetch_first_tile_ple(bundle, tokens);
        let vocab = bundle.config.vocab_size;
        let max_chunk = self.prompt_chunk_rows();
        let preflight_rows = if output_rows == Qwen4OutputRows::Final {
            (tokens.len() - 1) % max_chunk + 1
        } else {
            tokens.len().min(max_chunk)
        };
        let preflight_output_rows = output_rows.count(preflight_rows);
        let preflight_logits = logits.sub_offset(0, preflight_output_rows * vocab);
        self.preflight_output_resources(
            bundle,
            preflight_rows,
            &preflight_logits,
            preflight_output_rows,
        )?;
        let capture_width = wide_hidden_capture.map(|_| program_dims(&bundle.config).wide());
        let mut offset = 0usize;
        while offset < tokens.len() {
            // A planned pair (`on`, `auto` and `emulate` alike) runs layer-major
            // through the pair executor; everything else is an ordinary tile.
            self.poll_npu_trunk(gpu)?;
            #[cfg(target_os = "linux")]
            if let Some(banks) =
                self.npu_trunk_pair(bundle.state.position, tokens.len() - offset, max_chunk)
            {
                debug_assert!(
                    self.share.is_none(),
                    "the NPU trunk never runs under decode-share"
                );
                self.refuse_npu_pair()?;
                let rows = banks[0] + banks[1];
                self.note_npu_trunk_pair(banks, bundle.state.position);
                let final_pair =
                    self.set_tile_ple_lookahead(tokens, offset, rows, &mut caller_lookahead);
                let output = if output_rows == Qwen4OutputRows::All || final_pair {
                    Qwen4OutputPolicy::Rows(output_rows)
                } else {
                    Qwen4OutputPolicy::None
                };
                npu_forward::run_pair_request(
                    self,
                    bundle,
                    gpu,
                    &tokens[offset..offset + rows],
                    offset,
                    banks,
                    Some(logits),
                    top1,
                    wide_hidden_capture,
                    output,
                )?;
                offset += rows;
                continue;
            }
            let rows = (tokens.len() - offset).min(max_chunk);
            // The caller's lookahead belongs to the final tile; earlier tiles
            // look ahead to the next tile.
            let final_chunk =
                self.set_tile_ple_lookahead(tokens, offset, rows, &mut caller_lookahead);
            let selected_rows = output_rows.count(rows);
            let logits_offset = if output_rows == Qwen4OutputRows::All {
                offset
                    .checked_mul(vocab)
                    .ok_or_else(|| invalid("Qwen4 chunk logits offset overflows"))?
            } else {
                0
            };
            let logits_len = selected_rows
                .checked_mul(vocab)
                .ok_or_else(|| invalid("Qwen4 chunk logits shape overflows"))?;
            let logits_chunk = logits.sub_offset(logits_offset, logits_len);
            let top1_chunk = top1
                .filter(|_| output_rows == Qwen4OutputRows::All || final_chunk)
                .map(|destination| {
                    destination.sub_offset(
                        if output_rows == Qwen4OutputRows::All {
                            offset * std::mem::size_of::<i32>()
                        } else {
                            0
                        },
                        selected_rows * std::mem::size_of::<i32>(),
                    )
                });
            let capture_chunk = wide_hidden_capture
                .zip(capture_width)
                .map(|(capture, wide)| {
                    let capture_offset = offset
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture offset overflows"))?;
                    let capture_len = rows
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture shape overflows"))?;
                    Ok::<_, Qwen4GpuForwardError>(capture.sub_offset(capture_offset, capture_len))
                })
                .transpose()?;
            if let Some(layer_capture) = self.layer_capture.as_mut() {
                layer_capture.row = offset;
            }
            let output_policy = if output_rows == Qwen4OutputRows::All || final_chunk {
                Qwen4OutputPolicy::Rows(output_rows)
            } else {
                Qwen4OutputPolicy::None
            };
            self.forward_chunk_inner(
                bundle,
                gpu,
                &tokens[offset..offset + rows],
                None,
                Some(&logits_chunk),
                top1_chunk.as_ref(),
                capture_chunk.as_ref(),
                output_policy,
            )?;
            offset = offset
                .checked_add(rows)
                .ok_or_else(|| invalid("Qwen4 chunk offset overflows"))?;
        }
        Ok(())
    }

    /// Run bounded forward tiles that only commit model state: every tile
    /// takes [`Qwen4OutputPolicy::None`], so no head, logits or top-1 rows are
    /// produced. With `wide_hidden_capture` each tile still writes its wide
    /// hidden rows into the capture at `offset * wide` (the trunk's output,
    /// taken before the head, so identical to [`Self::forward_chunk`]'s).
    /// Tiles are `prompt_chunk_rows()` rows from the start of `tokens`, exactly
    /// as [`Self::forward_chunk`] tiles them, so a prefill split at tile
    /// multiples keeps the trunk numerics byte-identical.
    pub(crate) fn forward_chunk_silent(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        wide_hidden_capture: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.share_prompt_begin();
        let result = self.forward_chunk_silent_tiles(bundle, gpu, tokens, wide_hidden_capture);
        let finish = self.share_prompt_end(gpu, result.is_ok());
        merge_share_result(result, finish)
    }

    /// The tile loop of [`Self::forward_chunk_silent`].
    fn forward_chunk_silent_tiles(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        wide_hidden_capture: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let mut caller_lookahead = Some(std::mem::take(&mut self.ple_lookahead));
        self.validate_tokens(bundle, tokens)?;
        self.plan_npu_trunk_request(gpu, tokens.len(), bundle.state.position)?;
        let capture_width = wide_hidden_capture.map(|_| program_dims(&bundle.config).wide());
        if let Some((capture, wide)) = wide_hidden_capture.zip(capture_width) {
            let expected = tokens
                .len()
                .checked_mul(wide)
                .ok_or_else(|| invalid("Qwen4 wide hidden capture shape overflows"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        self.prefetch_first_tile_ple(bundle, tokens);
        let max_chunk = self.prompt_chunk_rows();
        let mut offset = 0usize;
        while offset < tokens.len() {
            self.poll_npu_trunk(gpu)?;
            #[cfg(target_os = "linux")]
            if let Some(banks) =
                self.npu_trunk_pair(bundle.state.position, tokens.len() - offset, max_chunk)
            {
                debug_assert!(
                    self.share.is_none(),
                    "the NPU trunk never runs under decode-share"
                );
                self.refuse_npu_pair()?;
                let rows = banks[0] + banks[1];
                self.note_npu_trunk_pair(banks, bundle.state.position);
                self.set_tile_ple_lookahead(tokens, offset, rows, &mut caller_lookahead);
                npu_forward::run_pair_request(
                    self,
                    bundle,
                    gpu,
                    &tokens[offset..offset + rows],
                    offset,
                    banks,
                    None,
                    None,
                    wide_hidden_capture,
                    Qwen4OutputPolicy::None,
                )?;
                offset += rows;
                continue;
            }
            let rows = (tokens.len() - offset).min(max_chunk);
            self.set_tile_ple_lookahead(tokens, offset, rows, &mut caller_lookahead);
            let capture_chunk = wide_hidden_capture
                .zip(capture_width)
                .map(|(capture, wide)| {
                    let capture_offset = offset
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture offset overflows"))?;
                    let capture_len = rows
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture shape overflows"))?;
                    Ok::<_, Qwen4GpuForwardError>(capture.sub_offset(capture_offset, capture_len))
                })
                .transpose()?;
            if let Some(layer_capture) = self.layer_capture.as_mut() {
                layer_capture.row = offset;
            }
            self.forward_chunk_inner(
                bundle,
                gpu,
                &tokens[offset..offset + rows],
                None,
                None,
                None,
                capture_chunk.as_ref(),
                Qwen4OutputPolicy::None,
            )?;
            offset = offset
                .checked_add(rows)
                .ok_or_else(|| invalid("Qwen4 chunk offset overflows"))?;
        }
        Ok(())
    }

    /// Device view of `tokens` as the ids the last forward holds at row
    /// `offset` of its `scratch.token_ids` upload, or `None` when that forward
    /// did not upload them whole from the host or they differ from `tokens`.
    /// The view borrows `scratch.token_ids`: the next forward overwrites it.
    pub(crate) fn uploaded_token_ids(&self, offset: usize, tokens: &[u32]) -> Option<GpuTensor> {
        let (byte_offset, byte_len) = uploaded_ids_range(
            self.uploaded_id_rows,
            &self.host_token_bytes,
            offset,
            tokens,
        )?;
        Some(view(&self.scratch.token_ids, byte_offset, byte_len))
    }

    /// Start the first tile's PLE rows now, unless a retained lookahead
    /// already names them. The tile's forward adopts this ticket (same epoch,
    /// same ids).
    fn prefetch_first_tile_ple(&mut self, bundle: &Qwen4Bundle, tokens: &[u32]) {
        // A profile without PLE has no rows to start.
        let Some(owner) = bundle.ple.as_ref() else {
            return;
        };
        let first_ids = bundle.state.ple_history.row_ids(
            &owner.metadata,
            &tokens[..tokens.len().min(self.prompt_chunk_rows())],
        );
        if !self
            .ple_ahead
            .as_ref()
            .is_some_and(|ticket| owner.rows.ticket_matches(ticket, &first_ids))
        {
            drop(self.ple_ahead.take());
            self.ple_ahead = owner
                .rows
                .prefetch(owner.rows.current_epoch(), first_ids)
                .ok();
        }
    }

    /// Name what follows the tile at `tokens[offset..offset + rows]`: the
    /// caller's lookahead for the final tile, the next tile's rows otherwise.
    /// Returns whether this is the final tile.
    fn set_tile_ple_lookahead(
        &mut self,
        tokens: &[u32],
        offset: usize,
        rows: usize,
        caller_lookahead: &mut Option<Vec<u32>>,
    ) -> bool {
        let final_chunk = offset + rows == tokens.len();
        if final_chunk {
            self.ple_lookahead = caller_lookahead.take().unwrap_or_default();
        } else {
            let next = offset + rows;
            self.ple_lookahead.clear();
            let window = self.prompt_chunk_rows();
            self.ple_lookahead
                .extend_from_slice(&tokens[next..(next + window).min(tokens.len())]);
        }
        final_chunk
    }

    /// Effective arithmetic/storage limit for eager verify. QSA follows the
    /// actual AR reference bound at window entry, including recorded AR; it
    /// does not impose an eager/recorded mismatch fallback. Nondefault MQ4
    /// singleton recipes without a matching rowtile remain explicitly narrow.
    pub(crate) fn guarded_wide_verify_rows(&self, bundle: &Qwen4Bundle, gpu: &Gpu) -> usize {
        let admitted = qwen4_row_exact_verify_rows(gpu);
        let recipe_rows = if gpu.qwen4_row_tile_reference_supported() {
            admitted
        } else {
            admitted.min(rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS)
        };
        let rows = recipe_rows.min(qwen4_spec_logit_rows(self.scratch.max_chunk));
        let eligible_single_fn = self.ep.is_none()
            && self.scope == Qwen4BandScope::single(&bundle.config)
            && bundle.config.profile.is_flash_next()
            && !self.lane_scope
            && self.qsa_tap.is_none()
            && self.mesh_route_rows.is_none()
            && self.cut_layers.is_none()
            && bundle.ep.is_none()
            && bundle.pp.is_none()
            && bundle.pp_band.is_none()
            && !hipfire_dispatch::pipeline::qsa_projection_hook_installed();
        qwen4_scoped_verify_rows(rows, eligible_single_fn, self.native_mtp_cpu_window_rows())
    }

    /// The singleton continuation's eligibility, not this multi-row call's.
    /// Keep the bundle terms in sync with the retained-body boundary below.
    pub(crate) fn ar_launch_reference(&self, bundle: &Qwen4Bundle, gpu: &Gpu) -> Qwen4ArLaunch {
        let (pm_invariant, append) = Self::route_scopes(false, gpu, 1);
        let retained_candidate = self.ep.is_none()
            && !pm_invariant
            && !append
            && !self.call_executes_host_arith(
                gpu,
                false,
                1,
                1,
                false,
                bundle.config.hidden_size,
                bundle.config.moe_intermediate_size,
                bundle.config.num_experts_per_tok,
            )
            && bundle.config.profile.is_flash_next()
            && !self.lane_scope
            && self.qsa_tap.is_none()
            && self.mesh_route_rows.is_none()
            && self.cut_layers.is_none()
            && !hipfire_dispatch::pipeline::qsa_projection_hook_installed();
        qwen4_ar_launch(
            retained_candidate,
            gpu.replay.request(),
            gpu.replay.state(),
            gpu.graphs.capture_mode,
        )
    }

    /// Number of MoE runtimes this forward holds.
    pub(crate) fn moe_count(&self) -> usize {
        self.moe.iter().flatten().count()
    }

    fn npu_trunk_loaded(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.npu_trunk.is_some()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// The band this forward runs.
    pub fn band_scope(&self) -> &Qwen4BandScope {
        &self.scope
    }

    /// Op ids of the band program a mesh forward runs, in issue order: the
    /// trunk ops (input `Embed` first on the input band), then on the output
    /// band the final HC read and LM head when `head`, and the argmax when
    /// `top1` (which requires `head`).
    pub fn mesh_op_range(&self, head: bool, top1: bool) -> std::ops::Range<u32> {
        let mut range = self.mesh_trunk_ops.clone();
        if self.scope.output && head {
            range.end += 2 + u32::from(top1);
        }
        range
    }

    /// The routing/scope flags a forward of `rows` rows runs under, from
    /// whether the bundle's prefill entry set `prefill_scope`.
    fn route_scopes(prefill_scope: bool, gpu: &Gpu, rows: usize) -> (bool, bool) {
        qwen4_route_scope_flags(
            prefill_scope,
            &gpu.arch,
            rows,
            rdna_compute::prefill_invariant::qwen4_append_wmma_enabled(),
            rdna_compute::prefill_invariant::qwen4_append_wmma_min_rows(),
            rdna_compute::prefill_invariant::prefill_pm_invariant_enabled(),
        )
    }

    /// Exact HC-streams geometry of a `rows`-row mesh forward on `route`:
    /// the canonical routing rows, the stream dtype and the live bytes. Pure
    /// of forward state; the same decision `forward_mesh_chunk` re-checks.
    pub fn mesh_layout(
        &self,
        bundle: &Qwen4Bundle,
        gpu: &mut Gpu,
        route: MeshRoute,
        rows: usize,
    ) -> Result<Qwen4MeshLayout, Qwen4GpuForwardError> {
        if self.share.is_some() {
            return Err(invalid(
                "a mesh forward cannot run on a decode-share forward (bank-selected scratch)",
            ));
        }
        if rows == 0 || rows > self.scratch.max_chunk {
            return Err(invalid(format!(
                "mesh forward of {rows} rows exceeds the configured capacity {}",
                self.scratch.max_chunk
            )));
        }
        let (pm_invariant, append) = Self::route_scopes(route == MeshRoute::Prefill, gpu, rows);
        let prior = std::mem::replace(&mut gpu.prefill_invariant_scope, pm_invariant || append);
        let route_rows = gpu.route_rows(rows);
        let bf16_streams = gpu.qwen4_bf16_streams(rows);
        gpu.prefill_invariant_scope = prior;
        let stream_bytes = rows
            .checked_mul(program_dims(&bundle.config).wide())
            .and_then(|elements| elements.checked_mul(if bf16_streams { 2 } else { 4 }))
            .ok_or_else(|| invalid("Qwen4 stream bytes overflow"))?;
        Ok(Qwen4MeshLayout {
            route_rows,
            bf16_streams,
            stream_bytes,
        })
    }

    /// The live HC streams of `layout` as an exact-byte view of
    /// `scratch.streams`: a non-input band receives into it, a non-output band
    /// publishes from it (its last `HyperWrite` leaves them there).
    pub fn mesh_stream_view(
        &self,
        layout: &Qwen4MeshLayout,
    ) -> Result<GpuTensor, Qwen4GpuForwardError> {
        let elements = layout.stream_bytes / std::mem::size_of::<f32>();
        if layout.stream_bytes % std::mem::size_of::<f32>() != 0
            || elements > self.scratch.streams.numel()
        {
            return Err(invalid("mesh stream bytes do not fit the stream scratch"));
        }
        Ok(view(&self.scratch.streams, 0, elements))
    }

    /// Refuse every route a mesh forward must not take: retained replay,
    /// the NPU trunk, expert staging/EP, lanes and developer taps.
    fn check_mesh_admission(
        &self,
        bundle: &Qwen4Bundle,
        gpu: &Gpu,
    ) -> Result<(), Qwen4GpuForwardError> {
        if self.scope != *bundle.weights.band_scope()
            || bundle.state.scope_map().band() != self.scope.band
        {
            return Err(invalid(
                "mesh forward bundle is not the band this forward was built for",
            ));
        }
        let refused = if self.ep.is_some() {
            Some("expert parallelism")
        } else if self.expert_stage.is_some() {
            Some("expert staging")
        } else if self.npu_trunk_loaded() {
            Some("the NPU trunk")
        } else if self.qsa_tap.is_some() {
            Some("the QSA parity tap")
        } else if self.route_trace.is_some() {
            Some("the routing trace")
        } else if self.cut_layers.is_some() {
            Some(CUT_LAYERS_ENV)
        } else if self.lane_scope {
            Some("lane scope")
        } else if !matches!(gpu.replay.state(), ReplayState::Hip | ReplayState::Fallback) {
            Some("a retained replay body")
        } else {
            None
        };
        match refused {
            Some(route) => Err(invalid(format!("a mesh forward refuses {route}"))),
            None => Ok(()),
        }
    }

    /// Map the QSA context (and the gathered-prefill workspace) a mesh
    /// forward through `end_position` needs, outside any mesh run: the run
    /// itself never grows context.
    pub fn prepare_mesh_capacity(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        end_position: usize,
        route_rows: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.check_mesh_admission(bundle, gpu)?;
        if end_position > bundle.state.max_seq_len {
            return Err(invalid(format!(
                "Qwen4 forward end position {end_position} exceeds context capacity {}",
                bundle.state.max_seq_len
            )));
        }
        bundle.release_session_for_growth(gpu, end_position);
        bundle
            .state
            .ensure_mapped_capacity(gpu, end_position)
            .map_err(|error| {
                invalid(format!(
                    "map Qwen4 QSA context through {end_position} tokens: {error}"
                ))
            })?;
        if route_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
            if let Some(qsa) = bundle
                .state
                .qsa
                .first()
                .filter(|qsa| qsa.f16_arena.is_none())
            {
                rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(
                    gpu,
                    qsa.format,
                    bundle.config.num_key_value_heads,
                    end_position,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue one band forward of a mesh epoch: `tokens` (read for every
    /// band's history/position commit; uploaded and embedded only by the input
    /// band) of canonical routing rows `route_rows` on `route`. A non-input
    /// band starts from the HC streams already received into
    /// `scratch.streams`; a non-output band leaves them there after its last
    /// `HyperWrite`; only the output band takes `head` (final HC collapse, LM
    /// head, argmax). Position, QSA marks and PLE history commit on every band.
    /// Nothing here waits on the GPU; capacity comes from
    /// [`Self::prepare_mesh_capacity`].
    ///
    /// `wide_hidden_capture` is the final pre-collapse wide hidden of this
    /// epoch's `tokens.len()` rows (`hc_count * hidden_size` F32 per row, the
    /// same capture [`Self::forward_chunk`] takes: the trunk's output HC
    /// streams, widened from BF16 when the epoch's streams are BF16). It is
    /// independent of `head`, so a silent prompt-fill chunk captures without
    /// computing logits. Only the output band owns the final streams: a
    /// non-output band refuses a capture. The caller passes the exact
    /// sub-view for this call's rows (a chunk of a longer capture is
    /// `capture.sub_offset(offset * wide, rows * wide)`).
    ///
    /// `MeshRoute::Verify` is the Single speculative verify forward
    /// (`spec_forward_rows`): every band runs outside the prefill scope with
    /// the epoch's actual rows as its routing rows, exactly like Decode but
    /// for `1..=max_chunk` rows. The output band must carry a head
    /// (verify logits); non-output bands carry none.
    ///
    /// `ops` is the segment of the band program (`mesh_op_range` of `head`)
    /// this call runs: the whole range for a PP band, the ops between two
    /// collectives for a TP rank. The head ops run with the last segment, so
    /// no segment may end inside them.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mesh_chunk(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        route_rows: usize,
        route: MeshRoute,
        head: Option<Qwen4MeshHead<'_>>,
        wide_hidden_capture: Option<&GpuTensor>,
        ops: std::ops::Range<u32>,
    ) -> Result<(), Qwen4GpuForwardError> {
        if self.share.is_some() {
            return Err(invalid(
                "a mesh forward cannot run on a decode-share forward (bank-selected scratch)",
            ));
        }
        self.check_mesh_admission(bundle, gpu)?;
        let full = self.mesh_op_range(head.is_some(), head.is_some_and(|head| head.top1.is_some()));
        let trunk = self.mesh_trunk_ops.clone();
        if ops.is_empty()
            || ops.start < full.start
            || ops.start > trunk.end
            || ops.end > full.end
            || (ops.end > trunk.end && ops.end != full.end)
        {
            return Err(invalid(format!(
                "mesh op range {ops:?} is not a segment of the band program {full:?}"
            )));
        }
        // Segments run in order, each once: a program starts at `full.start`
        // and continues where the previous segment ended.
        let expected = self.mesh_next_op.unwrap_or(full.start);
        if ops.start != expected {
            return Err(invalid(format!(
                "mesh op range {ops:?} does not continue the band program at op {expected}"
            )));
        }
        // Op ids map onto the trunk steps after the input band's `Embed`.
        let first_step = trunk.start + u32::from(self.scope.input);
        let step = |op: u32| (op.clamp(first_step, trunk.end) - first_step) as usize;
        let segment = MeshSegment {
            steps: step(ops.start)..step(ops.end),
            first: ops.start == full.start,
            last: ops.end == full.end,
        };
        let rows = tokens.len();
        if route == MeshRoute::Decode && rows != 1 {
            return Err(invalid("a mesh Decode forward carries exactly one row"));
        }
        if route == MeshRoute::Verify && self.scope.output && head.is_none() {
            return Err(invalid(
                "a mesh Verify forward on the output band carries a head for the verify logits",
            ));
        }
        if route == MeshRoute::Verify && self.scope.output && wide_hidden_capture.is_none() {
            // Every Single native-MTP verify captures the wide hidden, which
            // also keeps its 1-row case off the retained decode body. A
            // capture-less 1-row Verify could take that body in Single; the
            // mesh never replays, so it is refused instead of assumed exact.
            return Err(invalid(
                "a mesh Verify forward on the output band captures the wide hidden",
            ));
        }
        if wide_hidden_capture.is_some() && !self.scope.output {
            return Err(invalid(
                "only the output band takes a mesh wide hidden capture",
            ));
        }
        let output_policy = match (&head, self.scope.output) {
            (Some(head), true) => Qwen4OutputPolicy::Rows(head.rows),
            (None, _) => Qwen4OutputPolicy::None,
            (Some(_), false) => {
                return Err(invalid("only the output band takes a mesh head"));
            }
        };
        let layout = self.mesh_layout(bundle, gpu, route, rows)?;
        if layout.route_rows != route_rows {
            return Err(invalid(format!(
                "mesh epoch routing rows {route_rows} differ from this forward's {}",
                layout.route_rows
            )));
        }
        if route != MeshRoute::Prefill && layout.route_rows != rows {
            // Decode and Verify never canonicalize: their routing rows are
            // their actual rows (the same scope Single's non-prefill spec
            // forward runs under).
            return Err(invalid(format!(
                "a mesh {route:?} forward of {rows} rows must route its actual rows, got {}",
                layout.route_rows
            )));
        }
        self.validate_tokens(bundle, tokens)?;
        let prior = (self.prefill_scope, self.mesh_route_rows);
        self.prefill_scope = route == MeshRoute::Prefill;
        self.mesh_route_rows = Some(route_rows);
        self.mesh_segment = Some(segment);
        let result = self.forward_chunk_inner(
            bundle,
            gpu,
            tokens,
            None,
            head.map(|head| head.logits),
            head.and_then(|head| head.top1),
            wide_hidden_capture,
            output_policy,
        );
        (self.prefill_scope, self.mesh_route_rows) = prior;
        self.mesh_segment = None;
        // A failed segment abandons the program (nothing was committed).
        self.mesh_next_op = (result.is_ok() && ops.end != full.end).then_some(ops.end);
        result
    }

    /// Forget a band program left part way through (a reset after a failed
    /// epoch): the next mesh call starts a new program.
    pub(crate) fn abandon_mesh_program(&mut self) {
        self.mesh_next_op = None;
    }
    /// Set the Gpu routing scopes one forward of `rows` rows runs under and
    /// return the previous values (restore with [`Self::leave_tile_scopes`]):
    /// [`Gpu::qwen4_scope`] (admits the Qwen4-only MQ6 X-LDS defaults), the
    /// canonical-routing switch (see `rdna_compute::prefill_invariant`: set by
    /// the batch-invariant prefill profile or by the agent-turn append route,
    /// so every row-count routing decision of the forward sees the canonical
    /// chunk size while launches keep the real row count) and the ragged IU4
    /// trunk scope. The ordinary plan dedicates HcMixed/GdnZ/GdnA/GdnB to
    /// max_chunk rows and only `[0,n)` contains live tokens: compact scratch
    /// may alias live fields, so its in-place padding is never armed. Returns
    /// whether the append route applies. Shared by the chunk-major forward and
    /// every S1 layer-major tile so both see identical routing.
    pub(crate) fn enter_tile_scopes(&self, gpu: &mut Gpu, rows: usize) -> (TileScopes, bool) {
        let scope = std::mem::replace(&mut gpu.qwen4_scope, true);
        let (pm_invariant, append) = Self::route_scopes(self.prefill_scope, gpu, rows);
        if append {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "qwen4 append route: prefill tiles of {}..511 rows on the canonical (>= 512-row) routes, MoE F16 WMMA (HIPFIRE_QWEN4_APPEND_WMMA)",
                    rdna_compute::prefill_invariant::qwen4_append_wmma_min_rows()
                );
            });
        }
        let invariant = std::mem::replace(&mut gpu.prefill_invariant_scope, pm_invariant || append);
        let f16_moe = std::mem::replace(
            &mut gpu.qwen4_append_f16_moe_scope,
            crate::slowlink_sp::f16_moe_scope(append, self.slowlink.sp_f16_moe()),
        );
        let ragged_scope = native_trunk_prefill_scope(
            self.prefill_scope,
            gpu.qwen4_trunk_iu4,
            &gpu.arch,
            self.scratch.mode,
        );
        let ragged = std::mem::replace(&mut gpu.qwen4_trunk_iu4_prefill_scope, ragged_scope);
        (
            TileScopes {
                scope,
                invariant,
                f16_moe,
                ragged,
            },
            append,
        )
    }

    /// Restore the values [`Self::enter_tile_scopes`] replaced.
    pub(crate) fn leave_tile_scopes(gpu: &mut Gpu, saved: TileScopes) {
        gpu.qwen4_scope = saved.scope;
        gpu.prefill_invariant_scope = saved.invariant;
        gpu.qwen4_append_f16_moe_scope = saved.f16_moe;
        gpu.qwen4_trunk_iu4_prefill_scope = saved.ragged;
    }

    /// Bank and runtime selection, the only place decode-share state changes.
    /// Every tile of every route that reaches [`Self::forward_chunk_scoped`]
    /// passes here.
    ///
    /// Prompt-tile predicate ([`decode_share_prefill_tile`]): the call is a
    /// prefill tile when it is not a device-argmax decode forward
    /// (`argmax_of`), not a lane step (`lane_scope`) and not a mesh segment
    /// (`mesh_route_rows`), and carries at least
    /// [`QWEN4_DECODE_SHARE_RESIDENT_ROWS`] rows. A verify window
    /// (`!prefill_scope`, at most `QWEN4_VERIFY_MAX_ROWS` rows) is therefore
    /// never a prefill tile, and every other call (decode, verify, lanes,
    /// tiles below the resident rows) runs on the resident bank and the
    /// Resident runtimes. A prefill tile swaps the smallest class holding its
    /// rows into `scratch` (banks present) and reads the BorrowPrefill
    /// runtimes (`moe_borrow` present) for the tile, restoring both on every
    /// return path. `bind` mode has runtimes but no banks.
    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_inner(
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
        self.share_check()?;
        self.ep_ordinary_note_call(tokens.len());
        if self.ep_reserve.is_some() {
            return self.ep_reserve_forward_chunk(
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
        if self.share.is_none() && self.moe_borrow.is_empty() {
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
        let prefill_tile = self.share_prefill_tile(tokens, argmax_of);
        let class = if prefill_tile {
            self.share_class_for_tile(tokens.len())?
        } else {
            None
        };
        match class {
            Some(class) => {
                self.share_acquire(gpu, class)?;
                if let Some(share) = self.share.as_mut() {
                    // Inside a whole-prompt entry this entry now owns the
                    // dirty borrow and may continue it with its tail tiles.
                    if share.prompt_depth > 0 {
                        share.prompt_borrowed = true;
                    }
                    share.swap_in(&mut self.scratch, class);
                }
            }
            None if !prefill_tile && self.share.is_some() => self.share_require_resident(gpu)?,
            None => {}
        }
        self.borrow_runtimes_active = prefill_tile && !self.moe_borrow.is_empty();
        if self.borrow_runtimes_active {
            self.borrow_view = self.borrow_view_for_tile(class, tokens.len());
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
        self.borrow_runtimes_active = false;
        if let Some(share) = self.share.as_mut() {
            share.swap_out(&mut self.scratch);
        }
        if self.share.is_none() {
            return result;
        }
        let after = self.share_after_tile(gpu, result.is_ok());
        merge_share_result(result, after)
    }

    /// One tile on the bank selected in `scratch`: every forward runs under
    /// this call's routing scopes; a multi-row verify latches the singleton AR
    /// reference before installing them.
    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_tile(
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
        // Latch before installing this call's routing scopes: a multi-row
        // verify remains eager but must reproduce the singleton AR arithmetic
        // that would run at this exact boundary.
        let verify_reference = if !self.prefill_scope && tokens.len() >= 2 {
            use rdna_compute::prefill_invariant::Qwen4VerifyReference;
            Some(match self.ar_launch_reference(bundle, gpu) {
                Qwen4ArLaunch::Eager => Qwen4VerifyReference::Eager,
                Qwen4ArLaunch::Recorded => Qwen4VerifyReference::Recorded,
            })
        } else {
            None
        };
        #[cfg(feature = "lab")]
        if verify_reference.is_some() {
            self.last_verify_reference = verify_reference;
        }
        let (scopes, _append) = self.enter_tile_scopes(gpu, tokens.len());
        // Only eager verify widens the few-row routes. Its reference binding
        // selects the actual AR attention arithmetic without enabling replay.
        // Restore both host scopes on the single success/error return path.
        let wide_verify = if !self.prefill_scope && tokens.len() >= 2 {
            self.guarded_wide_verify_rows(bundle, gpu)
        } else {
            0
        };
        let prior_wide_verify = gpu.set_wide_verify_rows(wide_verify);
        // A verify forward's rows launchers may take the gfx1151 PM row kernels
        // (`HIPFIRE_FN_LANE_PM_ROWS`, each column byte-identical to the 1-row
        // incumbent), as the lane step does.
        let prior_rows_scope = gpu.lane_rows_scope;
        gpu.lane_rows_scope |= wide_verify > 0;
        let prior_verify_reference = gpu.set_qwen4_verify_reference(verify_reference);
        let result = self.forward_chunk_scoped(
            bundle,
            gpu,
            tokens,
            argmax_of,
            logits,
            top1,
            wide_hidden_capture,
            output_policy,
        );
        Self::leave_tile_scopes(gpu, scopes);
        gpu.set_wide_verify_rows(prior_wide_verify);
        gpu.lane_rows_scope = prior_rows_scope;
        gpu.set_qwen4_verify_reference(prior_verify_reference);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_scoped(
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
        // Host arithmetic is host synchronous: never a retained body, and an
        // eager forward. D1 and an explicit-mode D2 hold for the whole forward;
        // the automatic default's D2 only for the calls that execute it (a real
        // narrow non-prefill window), so wide verify and prefill keep their
        // old graph/replay behaviour.
        // Lab presence for the CPU experimental capacity: the existing
        // oracle override must be present AND this must be an eager verify
        // call (non-prefill scope, >= 2 rows). Ordinary prefill, decode and
        // captured/recorded calls never arm, whatever the GPU few-row limit is.
        let lab_armed =
            !self.prefill_scope && tokens.len() >= 2 && Self::cpu_lab_override_present(gpu);
        let host_arith_call = self.call_executes_host_arith(
            gpu,
            self.prefill_scope,
            tokens.len(),
            gpu.route_rows(tokens.len()),
            lab_armed,
            bundle.config.hidden_size,
            bundle.config.moe_intermediate_size,
            bundle.config.num_experts_per_tok,
        );
        self.ensure_cpu_splice_eager(gpu, host_arith_call)?;
        let cpu_deferred_experts = host_arith_call;
        // A mesh segment runs part of the program (see `MeshSegment`); every
        // other forward is one whole segment.
        let segment = self.mesh_segment.clone();
        let (first, last) = segment.as_ref().map_or((true, true), |s| (s.first, s.last));
        let ple_lookahead = if last {
            std::mem::take(&mut self.ple_lookahead)
        } else {
            Vec::new()
        };
        if argmax_of.is_some() && tokens.len() != 1 {
            return Err(invalid("an argmax-token Qwen4 forward carries one token"));
        }
        let n = tokens.len();
        if n == 0 || n > self.scratch.max_chunk {
            return Err(invalid(format!(
                "chunk length {n} exceeds configured capacity {}",
                self.scratch.max_chunk
            )));
        }
        self.scratch.note_forward_rows(n);
        // The router fault row is the absolute token position for a prefill / verify
        // chunk. A one-token decode forward may be captured or retained, which bakes
        // the launch's row base in, so it reports row 0 (position = decode step).
        gpu.set_router_row_base(if n > 1 { bundle.state.position } else { 0 });
        // Before anything can read a donor a failed staged forward left
        // overwritten (decode included).
        if let Some(stage) = self.expert_stage.as_ref() {
            stage.restore_blocking(gpu)?;
        }
        let config = bundle.config.clone();
        let band = self.scope.band.clone();
        // Only the first segment uploads and embeds the tokens and stages PLE.
        let (input, output) = (self.scope.input && first, self.scope.output);
        let mesh_route_rows = self.mesh_route_rows;
        let mesh_run = mesh_route_rows.is_some();
        if self.layer_capture.is_some() && (mesh_run || !input || !output) {
            return Err(invalid(
                "the DFlash2 layer capture runs on a whole single-card trunk only",
            ));
        }
        if (!input && argmax_of.is_some())
            || (!output
                && (output_policy.requested_rows(n).is_some() || wide_hidden_capture.is_some()))
        {
            return Err(invalid(format!(
                "band {}..{} cannot run an argmax token, head or capture it does not own",
                band.start, band.end
            )));
        }
        let end_position = bundle
            .state
            .position
            .checked_add(n)
            .ok_or_else(|| invalid("Qwen4 forward position overflows"))?;
        if end_position > bundle.state.max_seq_len {
            return Err(invalid(format!(
                "Qwen4 forward end position {end_position} exceeds context capacity {}",
                bundle.state.max_seq_len
            )));
        }
        if argmax_of.is_none()
            && tokens
                .iter()
                .copied()
                .any(|token| (token as usize) >= config.vocab_size)
        {
            return Err(invalid(
                "Qwen4 token id is outside the embedding vocabulary",
            ));
        }
        let dims = program_dims(&config);
        let requested_rows = output_policy.requested_rows(n);
        if let Some(requested_rows) = requested_rows {
            let logits = logits
                .ok_or_else(|| invalid("Qwen4 head output policy requires a logits tensor"))?;
            let expected_logits = requested_rows
                .checked_mul(config.vocab_size)
                .ok_or_else(|| invalid("logit shape overflow"))?;
            if logits.dtype != DType::F32 || logits.numel() != expected_logits {
                return Err(invalid(format!(
                    "logits must be F32 with {expected_logits} elements for {output_policy:?} output"
                )));
            }
            if let Some(top1) = top1 {
                let expected = requested_rows
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or_else(|| invalid("top1 shape overflow"))?;
                if top1.dtype != DType::Raw || top1.buf.size() < expected {
                    return Err(invalid(
                        "top1 output must be Raw with one i32 per requested output row",
                    ));
                }
            }
        } else if top1.is_some() {
            return Err(invalid("Qwen4 discarded output cannot receive top1 rows"));
        }
        if let Some(capture) = wide_hidden_capture.as_ref() {
            let expected = n
                .checked_mul(dims.wide())
                .ok_or_else(|| invalid("wide hidden capture shape overflow"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        // Map the QSA context rows this forward writes and reads (every
        // arena through `end_position`) and, for a chunk the gathered prefill
        // attention takes, its F16 workspace over the same prefix: here,
        // before any view, capture or record exists. Neither grows while a
        // capture or record is armed; a failure leaves the position
        // unpublished and already mapped pages owned.
        // Session snapshots yield to context growth (only at a growth
        // boundary; no query on the steady path).
        if mesh_run {
            if bundle.state.mapped_context_tokens() < end_position {
                return Err(invalid(format!(
                    "mesh forward needs the QSA context mapped through {end_position} tokens \
                     (prepare_mesh_capacity), {} are mapped",
                    bundle.state.mapped_context_tokens()
                )));
            }
        } else {
            bundle.release_session_for_growth(gpu, end_position);
            let mapped_before = bundle.state.mapped_context_tokens();
            bundle
                .state
                .ensure_mapped_capacity(gpu, end_position)
                .map_err(|error| {
                    invalid(format!(
                        "map Qwen4 QSA context through {end_position} tokens: {error}"
                    ))
                })?;
            if gpu.route_rows(n) >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
                if let Some(qsa) = bundle
                    .state
                    .qsa
                    .first()
                    .filter(|qsa| qsa.f16_arena.is_none())
                {
                    rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(
                        gpu,
                        qsa.format,
                        config.num_key_value_heads,
                        end_position,
                    )?;
                }
            }
            if bundle.state.mapped_context_tokens() != mapped_before {
                if let Ok((target, mtp)) = bundle.qsa_context_committed_bytes(gpu) {
                    eprintln!(
                    "  qwen4 QSA context mapped for {} tokens: {} MiB target + {} MiB MTP committed, {} MiB gather workspace",
                    bundle.state.mapped_context_tokens(),
                    target >> 20,
                    mtp >> 20,
                    gpu.qsa_gather_scratch_bytes() >> 20
                );
                }
            }
        }
        // Without PLE (`config.ple` is `None`) there is no PLE layer, no row
        // geometry and no row fetch below.
        let ple_layer_index = if config.ple.is_some() {
            let index = config
                .ple_layer_ids
                .first()
                .and_then(|id| id.checked_sub(1))
                .ok_or_else(|| invalid("PLE layer id missing"))?;
            if index >= config.num_hidden_layers {
                return Err(invalid("PLE layer id is outside the trunk"));
            }
            Some(index)
        } else {
            None
        };
        let (ple_head_count, ple_row_width) = config
            .ple
            .as_ref()
            .map_or((0, 0), |ple| (ple.head_count, ple.row_width));
        let final_hyper = if requested_rows.is_some() {
            Some(hyper_read_desc(
                &bundle.weights,
                &bundle.weights.root.final_hyper,
            )?)
        } else {
            None
        };
        let lm_head = if requested_rows.is_some() {
            Some(dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?)
        } else {
            None
        };
        if let Some(requested_rows) = requested_rows {
            let final_hyper = final_hyper
                .as_ref()
                .ok_or_else(|| invalid("Qwen4 final hyper preflight disappeared"))?;
            let lm_head = lm_head
                .as_ref()
                .ok_or_else(|| invalid("Qwen4 LM head preflight disappeared"))?;
            let logits = logits
                .ok_or_else(|| invalid("Qwen4 logits output disappeared before preflight"))?;
            let preflight_scratch = layer_scratch(&self.scratch, &self.scratch.router_logits);
            validate_final_hyper(
                dims,
                final_hyper,
                &self.scratch.streams,
                &preflight_scratch,
                n,
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 final hyper: {error:?}"))
            })?;
            let hidden_elements = n
                .checked_mul(config.hidden_size)
                .ok_or_else(|| invalid("Qwen4 final hidden shape overflow"))?;
            let final_hidden = view(&self.scratch.hc_mixed, 0, hidden_elements);
            validate_lm_head(lm_head, &final_hidden, logits, n, requested_rows).map_err(
                |error| {
                    Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 LM head: {error:?}"))
                },
            )?;
        }

        // The PLE row fetch runs on the row store's worker while the step
        // program is built below; a matching retained lookahead is adopted.
        let retained = if input { self.ple_ahead.take() } else { None };
        let mut ple = match bundle.ple.as_ref().filter(|_| input) {
            None => {
                drop(retained);
                None
            }
            Some(owner) => Some(
                if argmax_of.is_none() {
                    RowFetch::begin_or_adopt(
                        &owner.rows,
                        retained,
                        bundle.state.ple_history.row_ids(&owner.metadata, tokens),
                    )
                } else {
                    drop(retained);
                    RowFetch::begin(&owner.rows)
                }
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?,
            ),
        };
        let router_logits = matrix_view(&self.scratch.router_logits, n, router_columns(dims))?;
        let scratch_desc = layer_scratch(&self.scratch, &router_logits);
        // Decided once: every op touching the HC streams in this forward agrees.
        let bf16_state = gpu.qwen4_bf16_streams(n);
        // The row count every routing decision below sees (`n` unless the
        // invariant prefill profile is on); launch extents keep `n`.
        let route_n = gpu.route_rows(n);
        // Largest row count of a few-row (decode-exact) route in this forward:
        // the historic 8, or the verify scope's row-exact rows.
        let few_row_limit = gpu.few_row_limit();
        if mesh_route_rows.is_some_and(|expected| expected != route_n) {
            return Err(invalid(
                "mesh epoch routing rows differ from the Gpu's routing scope",
            ));
        }
        // Expert parallelism picks the MoE table per forward by routing row
        // count, identically on every rank: one row, or a few-row forward
        // (`2..=QWEN4_EP_GATHER_MAX_ROWS` rows that are also the routing rows:
        // verify / replay / takeover), runs the rank's decode table with the
        // row gather installed and a parity gather buffer as the expanded-row
        // scratch; larger chunks run the full (P2P) table, no gather.
        let ep = self.ep.as_ref();
        let ep_few_rows =
            ep.is_some() && (2..=QWEN4_EP_GATHER_MAX_ROWS).contains(&route_n) && route_n == n;
        if let Some(ep) = ep {
            if route_n == 1 {
                if n != 1 {
                    return Err(invalid("an EP decode forward carries exactly one row"));
                }
                if ep.gather.is_none() {
                    return Err(invalid(
                        "EP decode forward before attach_ep: the row gather is not installed",
                    ));
                }
            } else if ep_few_rows {
                if ep.gather.is_none() {
                    return Err(invalid(
                        "EP few-row forward before attach_ep: the row gather is not installed",
                    ));
                }
            } else if ep.moe_full.len() != self.moe_count() {
                return Err(invalid(
                    "EP chunk forward before attach_ep: the full expert tables are not built",
                ));
            }
        }
        let ep_gathered = route_n == 1 || ep_few_rows;
        let ep_gather: Option<&dyn MoeRowGather> = ep.filter(|_| ep_gathered).and_then(|ep| {
            ep.gather
                .as_deref()
                .map(|gather| gather as &dyn MoeRowGather)
        });
        let ep_scratches: Option<[Qwen4LayerScratch<'_>; QWEN4_EP_GATHER_BUFFERS]> =
            ep.filter(|_| ep_gathered).map(|ep| {
                std::array::from_fn(|parity| Qwen4LayerScratch {
                    moe_down_expanded: &ep.buffers[parity],
                    ..layer_scratch(&self.scratch, &router_logits)
                })
            });
        // Owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`, hook installed by
        // the mesh): a chunk on the owner-routed route (symmetric IU4 on gfx1201, the F16
        // WMMA grouped route on gfx1100; >= 512 routing rows that are the launch rows) binds
        // each layer's compact table plus the tile exchange; every other chunk keeps the
        // full-table route exactly as without the lever.
        let tile_route = ep
            .filter(|_| route_n > 1 && !ep_few_rows)
            .and_then(|ep| ep.tile.as_ref())
            .filter(|route| route.hook.is_some());
        let tile_active = ep_reserve_forward::ep_tile_active(ep, gpu, &config, n, route_n);
        if let Some(route) = tile_route.filter(|_| !tile_active) {
            // A full-table chunk also writes `y_down_grouped`: it may start only after the
            // peer finished pulling this rank's rows of the last exchange.
            if let Some(hook) = route.hook.as_ref() {
                hook.before_down(gpu)
                    .map_err(|error| invalid(format!("EP tile exchange: {error}")))?;
            }
        }
        let tile_route = tile_route.filter(|_| tile_active);
        // G2 expert stage under EP: a full-table chunk reads another rank's donor VRAM, so every
        // rank drains its stage donors and crosses a host barrier before ANY rank launches it.
        if ep.is_some() && route_n > 1 && !ep_few_rows && !tile_active {
            // A stage is only ever USED (donors overwritten) on forwards that bind the compact
            // route, which needs the installed exchange; without its hook no forward staged, so
            // there is nothing to drain.
            let hook = ep
                .and_then(|ep| ep.tile.as_ref())
                .and_then(|route| route.hook.as_ref());
            if let (Some(stage), Some(hook)) = (self.expert_stage.as_ref(), hook) {
                stage.ep_quiesce(gpu)?;
                hook.host_barrier(
                    hipfire_runtime::ep_tile_exchange::HOST_STAGE_QUIESCE,
                    self.ep_reserve_epoch(),
                )
                .map_err(|error| invalid(format!("EP stage quiesce barrier: {error}")))?;
            }
        }
        if config.num_hidden_layers.saturating_mul(7).saturating_add(1) > QWEN4_STEP_INLINE_CAPACITY
        {
            return Err(invalid(
                "Qwen4 generic layer program inline capacity exhausted",
            ));
        }
        // Decode, small chunks and MTP's few-row wide-hidden verify keep
        // mapped reads (below `min_rows`). A long forward that captures wide
        // hidden rows (the MTP prompt fill) stages unless
        // `HIPFIRE_QWEN4_EXPERT_STAGE_CAPTURE=0`: the capture is a copy of
        // the output streams after the program, on the compute stream. The
        // parity observer keeps its historical unsplit transport. A stream
        // capture keeps mapped reads too: a staged forward forks a copy
        // stream and its failure recovery synchronizes the device.
        let expert_stage = match self
            .expert_stage
            .as_ref()
            .filter(|stage| stage.begin_forward(tile_active))
            .filter(|_| self.ep_reserve.is_none() || self.borrow_runtimes_active)
        {
            Some(stage)
                if n >= stage.min_rows
                    && (stage.capture
                        || (wide_hidden_capture.is_none() && self.layer_capture.is_none()))
                    && self.qsa_tap.is_none() =>
            {
                let capturing = match gpu.active_stream.as_ref() {
                    Some(stream) => gpu.hip.stream_is_capturing(stream)?,
                    None => false,
                };
                (!capturing).then_some(stage)
            }
            _ => None,
        };
        if let Some(stage) = expert_stage {
            // Slab stage: the first two host layers copy while the resident
            // layers compute (the legacy donors refill from their own MoE).
            stage.slab_prime(gpu, self.tile_moe().len())?;
        }
        // The runtimes this tile reads, as a slice borrowed from the two
        // runtime fields only (`Self::tile_moe` would borrow all of `self`
        // across the program build and its closures).
        let tile_moe = tile_moe_of(
            self.borrow_runtimes_active,
            self.borrow_view,
            &self.moe,
            &self.moe_borrow,
        );
        // Whether the donor tail is whole for this call (read once, before the
        // program closures borrow `self`; the loan state cannot change inside a tile).
        let donor_resident = self.share_state_resident();
        // The MoE seal reads this call's canonical route rows from the
        // dispatch context (`batch_size = route_rows`): a short tail of a long
        // prefill keeps `rows != route_rows` and the GPU route, while a whole
        // logical verify window (`rows == route_rows`) may take the qualified
        // CPU path. Ordinary prefill chunk seals keep `route_rows` = chunk.
        // Per-call actual eager-verify scope for the CPU-owned expert window:
        // global override presence alone MUST NOT arm it (a whole 64-row
        // ordinary prefill has `rows == route_rows` too). Only a real eager
        // verify call sets it: non-prefill scope, `>= 2` rows, neither
        // recording nor graph-capturing, override present.
        let cpu_wide_verify = !self.prefill_scope
            && n >= 2
            && !gpu.replay.is_recording()
            && !gpu.graphs.capture_mode
            && Self::cpu_lab_override_present(gpu);
        let ctx = DispatchCtx::new(gpu)
            .with_batch_size(route_n)
            .with_cpu_wide_verify(cpu_wide_verify);
        // Heap, not an inline SmallVec: 384 inline steps are ~567 KiB, and the `attempt`
        // closure below moves `steps`, so the frame held three copies (~1.7 MiB). That
        // overflowed the 2 MiB stack of an EP peer rank's worker thread in a chunk forward.
        let mut steps: Vec<Step<'_>> = Vec::with_capacity(QWEN4_STEP_INLINE_CAPACITY);
        // `HIPFIRE_QWEN4_PLE_FUSE`: the PLE rows are staged straight as F16 for
        // the BF16 key/value projections (decided once; the step op and the
        // staging closure agree on it).
        let mut ple_rows_f16 = false;
        // `HIPFIRE_CHUNK_DUMP_DIR` (developer diagnostic; off by default).
        let chunk_dump = crate::chunk_dump::config().filter(|_| route_n > 1);
        let mut dump_ends: Vec<(usize, usize, bool)> = Vec::new();
        // `HIPFIRE_QWEN4_CUT_LAYERS`: step index at which each listed layer starts.
        let cut_layers = self.cut_layers;
        let mut cut_steps = [0usize; MAX_CUT_LAYERS];
        let mut cut_count = 0usize;
        // One typed per-layer step builder serves this chunk-major program and the S1
        // layer-major prompt schedule (`crate::slowlink_prefill`): the phase methods of
        // `LayerPhaseEnv` are the only place those steps are lowered.
        let layer_env = LayerPhaseEnv {
            scratch: &self.scratch,
            decode_q8: &self.decode_q8,
            trunk_zba: &self.trunk_zba,
            trunk_a4: &self.trunk_a4,
            prefill_scope: self.prefill_scope,
            bundle: &*bundle,
            dims,
            bf16_state,
            route_n,
            few_row_limit,
            rows: n,
            start_position: bundle.state.position,
            streams: &self.scratch.streams,
            gates: &self.scratch.hc_gates,
        };
        // DFlash2 layer capture: each captured layer's slot, keyed by the
        // step (the next layer's mixer) before which the streams hold its
        // completed output; the trunk's last layer is captured after the
        // program instead.
        let capture_layers: SmallVec<[usize; 8]> = self
            .layer_capture
            .as_ref()
            .map(|capture| capture.layers.clone())
            .unwrap_or_default();
        let mut capture_at: SmallVec<[(usize, usize); 8]> = SmallVec::new();
        for layer_index in band.clone() {
            if let Some(cuts) = cut_layers.as_ref() {
                if cuts.contains(layer_index) {
                    cut_steps[cut_count] = steps.len();
                    cut_count += 1;
                }
            }
            let layer = bundle
                .weights
                .layer(layer_index)
                .ok_or_else(|| invalid(format!("layer {layer_index} is not owned by this band")))?;
            if ple_layer_index == Some(layer_index) {
                let ple_weights = layer
                    .ple
                    .as_ref()
                    .ok_or_else(|| invalid("PLE weights missing"))?;
                if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
                    return Err(invalid(
                        "Qwen4 generic layer program inline capacity exhausted",
                    ));
                }
                let (ple_step, rows_f16) =
                    layer_env.ple_step(gpu, ple_weights, ple_head_count, ple_row_width)?;
                ple_rows_f16 = rows_f16;
                steps.push(ple_step);
            }

            // Balanced ownership: a host-mapped trunk layer of a gathered EP
            // forward (one row, or few rows) binds its live runtime, whose
            // tables the layer's owner assignment rewrites after routing.
            // Every other layer keeps the static EP table (or the full table
            // for chunk forwards). `ep.balanced` is empty unless the env
            // lever was on at attach, so off binds exactly what it always did.
            let balanced_live = ep
                .filter(|_| ep_gathered)
                .and_then(|ep| ep.balanced.get(layer_index))
                .and_then(Option::as_ref)
                .map(|balanced| &balanced.live);
            let layer_moe = select_layer_moe(
                balanced_live,
                ep.map(|ep| ep.moe_full.get(layer_index)),
                route_n > 1 && !ep_few_rows,
                tile_route.is_some(),
                tile_moe.get(layer_index).and_then(Option::as_ref),
            );
            if let Some(stage) = expert_stage {
                stage.prepare_runtime(gpu, layer_index, layer_moe)?;
            }
            let mut description = layer_desc(
                &bundle.weights,
                layer,
                layer_moe,
                &config,
                &self.decode_q8,
                route_n,
                few_row_limit,
                // Staged expert tables never replace the live table.
                expert_stage.filter(|_| balanced_live.is_none()),
            )?;
            if let (Some(gather), Qwen4FfnBinding::Moe(binding)) = (ep_gather, &mut description.ffn)
            {
                binding.row_gather = Some(gather);
                if ep_few_rows {
                    // The gather buffer is the combine's `down_expanded`; the
                    // unfused shared down writes the ordinary scratch.
                    binding.shared_down_out = Some(&self.scratch.moe_down_expanded);
                }
            }
            if let (Some(route), Qwen4FfnBinding::Moe(binding)) = (tile_route, &mut description.ffn)
            {
                if let Some(hook) = route.hook.as_deref() {
                    binding.tile_exchange = Some(MoeTileExchangeParams {
                        exchange: hook,
                        tile_ids_owned: &route.tile_ids_owned,
                    });
                }
            }
            // DFlash2 layer capture of the previous layer: keyed by this
            // layer's mixer, the step after `push_mixer_prefix`'s HC read.
            let captured_slot = layer_index
                .checked_sub(1)
                .and_then(|previous| capture_layers.iter().position(|&l| l == previous));
            if let Some(slot) = captured_slot {
                if ple_layer_index == Some(layer_index) {
                    return Err(invalid(format!(
                        "DFlash2 layer capture of layer {} would include the PLE layer",
                        layer_index - 1
                    )));
                }
                capture_at.push((steps.len() + 1, slot));
            }
            layer_env.push_mixer_prefix(&mut steps, layer_index, layer, &description)?;

            match &description.ffn {
                Qwen4FfnBinding::Moe(moe) => {
                    // The D2 executor gate sees the whole routing window plus
                    // this call's qualified width and allocated capacity, so a
                    // call whose real rows are not its whole routing window (a
                    // short tail of a long prefill) or that is wider than the
                    // qualified cap is bound to a view of the runtime this
                    // layer actually selected (`layer_moe`: balanced live, EP
                    // full chunk, or static) with no executor. The view must be
                    // of that same runtime: its table and cache were bound to
                    // it, and the seal compares them. An admitted whole-window
                    // 1..64-row verify keeps one forward with full-window
                    // state/capture, never chunked.
                    let mut moe = *moe;
                    let verify_rows = qwen4_row_exact_verify_rows(gpu);
                    let lab_armed =
                        !self.prefill_scope && n >= 2 && Self::cpu_lab_override_present(gpu);
                    moe.routed_experts =
                        bind_routed_view(layer_moe, moe.routed_experts, |runtime| {
                            runtime.routed_view(
                                !self.prefill_scope,
                                n,
                                route_n,
                                lab_armed,
                                verify_rows,
                            )
                        });
                    layer_env.moe_steps(
                        &mut steps,
                        &ctx,
                        moe,
                        ep_scratches.as_ref().map_or(&scratch_desc, |parity| {
                            &parity[qwen4_ep_buffer_index(layer_index, config.num_hidden_layers)]
                        }),
                    )?;
                }
                Qwen4FfnBinding::Dense(dense) => {
                    steps.push(layer_env.dense_step(dense));
                }
            }

            steps.push(layer_env.mlp_write_step(&description, &self.scratch.ffn_output));
            if chunk_dump.is_some_and(|dump| dump.wants(layer_index)) {
                let bf16_grouped = layer_moe.is_some_and(|moe| moe.symmetric)
                    && gpu.qwen4_moe_sym_iu4_applies(route_n);
                dump_ends.push((layer_index, steps.len(), bf16_grouped));
            }
        }
        let capture_last = capture_layers
            .iter()
            .position(|&layer| layer + 1 == band.end);
        if capture_at.len() + usize::from(capture_last.is_some()) != capture_layers.len()
            || capture_layers.len() > 32
        {
            return Err(invalid(format!(
                "DFlash2 layer capture {capture_layers:?} is outside trunk layers {}..{}",
                band.start, band.end
            )));
        }
        validate_steps(gpu, &steps).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("preflight Qwen4 typed program: {error:?}"))
        })?;
        // A retained replay does not execute host code inside its steps. Derive
        // the same QSA commit for either route before any forward-side effects.
        let mut qsa_commits: SmallVec<
            [(usize, usize, usize, usize, usize, usize); QWEN4_QSA_INLINE_CAPACITY],
        > = SmallVec::new();
        for step in &steps {
            if let Step::IndexedAttention(op) = step {
                if qsa_commits.len() >= QWEN4_QSA_INLINE_CAPACITY {
                    return Err(invalid("Qwen4 QSA commit inline capacity exhausted"));
                }
                let (full_len, raw_len, pooled_len, selected_len, position) =
                    op.next_lengths().map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!("derive Qwen4 QSA state: {error:?}"))
                    })?;
                qsa_commits.push((
                    qsa_commits.len(),
                    full_len,
                    raw_len,
                    pooled_len,
                    selected_len,
                    position,
                ));
            }
        }
        // A mesh segment executes only its steps; the QSA commit above is of
        // the whole program and lands with the last segment.
        if let Some(segment) = segment.as_ref().filter(|s| !(s.first && s.last)) {
            if segment.steps.end > steps.len() || (segment.last && segment.steps.end != steps.len())
            {
                return Err(invalid(format!(
                    "mesh segment steps {:?} do not fit the {}-step band program",
                    segment.steps,
                    steps.len()
                )));
            }
            if !dump_ends.is_empty() {
                return Err(invalid(
                    "the chunk dump (HIPFIRE_CHUNK_DUMP_DIR) cannot run a mesh segment",
                ));
            }
            steps.truncate(segment.steps.end);
            steps.drain(..segment.steps.start);
        }
        // A HIP body: no retained tape records or routes this forward.
        let hip_body = n == 1
            && !self.lane_scope
            && !mesh_run
            && !gpu.prefill_invariant_scope
            && matches!(gpu.replay.state(), ReplayState::Hip | ReplayState::Fallback);
        // `argmax_of` on a HIP body: the GPU argmax writes the token id and the
        // host reads it back only at the PLE layer, so the GPU runs this
        // forward's first layers while the host waits. Otherwise the host
        // takes the argmax now (the program above is built and nothing has
        // touched the device or the state yet).
        let device_token = argmax_of.filter(|_| hip_body);
        let host_argmax: [u32; 1];
        let tokens = match argmax_of {
            Some(previous) if !hip_body => {
                let token = rdna_compute::tensor_ops::argmax_f32_host(gpu, previous)?;
                if token as usize >= config.vocab_size {
                    return Err(invalid(
                        "Qwen4 token id is outside the embedding vocabulary",
                    ));
                }
                host_argmax = [token];
                if let (Some(ple), Some(owner)) = (ple.as_mut(), bundle.ple.as_ref()) {
                    ple.prefetch(
                        bundle
                            .state
                            .ple_history
                            .row_ids(&owner.metadata, &host_argmax),
                    )
                    .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                }
                &host_argmax[..]
            }
            _ => tokens,
        };
        // A queued early id copy still reads `token_ids`: drain it before this
        // forward (or its HIP-body fill) overwrites them.
        if let Some(readback) = self.token_readback.as_mut() {
            readback.complete(&gpu.hip)?;
        }
        let id_bytes = if input {
            n * std::mem::size_of::<i32>()
        } else {
            0
        };
        let ids = view(&self.scratch.token_ids, 0, id_bytes);
        // No claim on the device ids until this forward's own upload lands.
        self.uploaded_id_rows = 0;
        if let Some(previous) = device_token {
            argmax_f32(
                gpu,
                &ArgmaxF32 {
                    logits: previous,
                    indices: &ids,
                    rows: 1,
                    vocab: config.vocab_size,
                },
            )?;
            if self.token_readback.is_none() {
                self.token_readback =
                    Some(TokenReadback::create(&gpu.hip, self.token_readback_early)?);
            }
            if let Some(readback) = self.token_readback.as_mut() {
                if let Err(error) = readback.enqueue(
                    &gpu.hip,
                    &self.scratch.token_ids.buf,
                    gpu.active_stream.as_ref(),
                ) {
                    // The copy may already be queued (marked pending before
                    // submission): retire it, keeping the enqueue error.
                    let _ = readback.complete(&gpu.hip);
                    return Err(error.into());
                }
            }
        } else if !input {
            // A non-input band receives HC streams, not token ids.
        } else if mesh_run {
            // Fenced async upload: no host wait on the device queue.
            let ids_bytes = n * std::mem::size_of::<i32>();
            let staged_ids = self.mesh_tokens.stage_buffer(gpu, ids_bytes)?;
            for (index, token) in tokens.iter().copied().enumerate() {
                staged_ids[index * 4..index * 4 + 4].copy_from_slice(&(token as i32).to_ne_bytes());
            }
            self.mesh_tokens
                .upload_async(gpu, &self.scratch.token_ids.buf, ids_bytes)?;
        } else {
            for (index, token) in tokens.iter().copied().enumerate() {
                let bytes = &mut self.host_token_bytes[index * 4..index * 4 + 4];
                bytes.copy_from_slice(&(token as i32).to_ne_bytes());
            }
            if hip_body {
                // The one id is a stream-ordered fill, not a synchronous copy.
                let active = gpu.active_stream.as_ref();
                gpu.hip.memset_d32_async(
                    &self.scratch.token_ids.buf,
                    tokens[0] as i32,
                    1,
                    active,
                )?;
            } else {
                gpu.memcpy_htod_auto(
                    &self.scratch.token_ids.buf,
                    &self.host_token_bytes[..n * std::mem::size_of::<i32>()],
                )?;
                // Whole-row host upload on the default stream: the device ids
                // stay as uploaded until the next forward writes them.
                if !gpu.graphs.capture_mode {
                    self.uploaded_id_rows = n;
                }
            }
        }
        let embed_len = if input { n * config.hidden_size } else { 0 };
        let embedding_rot = view(&self.scratch.embedding_rot, 0, embed_len);
        let embeddings = view(&self.scratch.embeddings, 0, embed_len);
        // Every fallible step between the readback enqueue and the `attempt`
        // closure: on failure retire the queued id copy (keeping the original
        // error) so a caller's lane/NPU path may rewrite `token_ids`.
        let embedded = (|| -> Result<(), Qwen4GpuForwardError> {
            if !input {
                return Ok(());
            }
            let embedding = bundle.weights.resident(&bundle.weights.root.embedding)?;
            execute_embedding(
                gpu,
                &EmbeddingOp {
                    table: embedding,
                    rotated: &embedding_rot,
                    token_ids: &ids,
                    output: &embeddings,
                    rows: n,
                    dim: config.hidden_size,
                },
            )
            .map_err(|e| Qwen4GpuForwardError::Dispatch(format!("Qwen4 embedding: {e:?}")))?;
            Ok(())
        })();
        if let Err(error) = embedded {
            if let Some(readback) = self.token_readback.as_mut() {
                let _ = readback.complete(&gpu.hip);
            }
            return Err(error);
        }

        let next_history = bundle.state.ple_history;
        let next_position = bundle.state.position;
        // Boundary samples, hoisted so the post-body finalize outside this closure
        // can report them: they are taken where the window opens.
        let mut device_read: Option<u32> = None;
        let mut diagnostic_capture = false;
        let mut traced = false;
        let attempt = (|| -> Result<
            SmallVec<[(usize, usize, usize, usize, usize, usize); QWEN4_QSA_INLINE_CAPACITY]>,
            Qwen4GpuForwardError,
        > {
            let ple_len = if input { n * ple_head_count * ple_row_width } else { 0 };
            let ple_rows = view(&self.scratch.ple_rows, 0, ple_len);
            let staged = view(&self.scratch.ple_staged, 0, ple_len);

            // Expand the embedding batch into the row-major stream basis with
            // one broadcast kernel.  The shared HC program consumes
            // [row, branch, hidden], so a launch-per-row/branch memcpy loop is
            // both unnecessary and visible in short AR prefill profiles.
            let wide = dims.wide();
            if input {
            if bf16_state {
                gpu.hc_streams_init_from_embed_batched_bf16(
                    &embeddings,
                    &self.scratch.streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    n as i32,
                )?;
            } else {
                gpu.hc_streams_init_from_embed_batched(
                    &embeddings,
                    &self.scratch.streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    n as i32,
                )?;
            }
            }

            // PLE rows: wait for the host fetch, stage, upload and widen them.
            // A prefill defers this until layer 0 is enqueued (the rows are first
            // read at the PLE layer), so the host fetch overlaps GPU work; a
            // single-token forward keeps it ahead of the retained-body boundary.
            let ple_host = &mut self.host_ple;
            let ple_split = steps
                .iter()
                .position(|step| matches!(step, Step::GroupedDepthwise(_)))
                .filter(|_| n > 1 || device_token.is_some());
            let deferred = ple_split.is_some();
            let mut readback = self.token_readback.as_mut();
            let token_ids = &self.scratch.token_ids;
            let history = &bundle.state.ple_history;
            let ple_metadata = bundle.ple.as_ref().map(|owner| &owner.metadata);
            let vocab = config.vocab_size;
            let read_token = &mut device_read;
            let mut stage_ple = |gpu: &mut Gpu| -> Result<(), Qwen4GpuForwardError> {
                if ple.as_ref().is_some_and(|ple| ple.lease().is_some()) {
                    return Ok(());
                }
                if device_token.is_some() {
                    // The argmax this forward enqueued first: wait for it alone
                    // and read its id on a stream independent of the layers
                    // enqueued since.
                    let readback = readback
                        .as_deref_mut()
                        .ok_or_else(|| invalid("Qwen4 token readback missing"))?;
                    let token = readback.read_token(&gpu.hip, &token_ids.buf)?;
                    gpu.take_device_wait_error()?;
                    if token as usize >= vocab {
                        return Err(invalid(
                            "Qwen4 token id is outside the embedding vocabulary",
                        ));
                    }
                    *read_token = Some(token);
                    if let (Some(ple), Some(ple_metadata)) = (ple.as_mut(), ple_metadata) {
                        ple.prefetch(history.row_ids(ple_metadata, &[token]))
                            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                    }
                }
                // A profile without PLE has no rows to wait for, stage, upload
                // or gather (the token read above is all this forward needs).
                let Some(ple) = ple.as_mut() else {
                    return Ok(());
                };
                // After a device-token readback the host has synchronized past
                // the previous forward's upload, so the staging buffer is free.
                let mode = if device_token.is_some() {
                    PleUploadMode::Unfenced
                } else if mesh_run && !gpu.graphs.capture_mode {
                    // Mesh: never block the host on the device queue; the fence
                    // keeps the staging bytes unwritable until the copy ran.
                    PleUploadMode::Async
                } else if ple_async_upload(
                    ple_host.enabled(),
                    false,
                    deferred,
                    gpu.graphs.capture_mode,
                ) {
                    // Deferred past layer 0: do not make the host wait for the
                    // layers already enqueued. The fence keeps the staging
                    // bytes unwritable until the copy has run.
                    PleUploadMode::Async
                } else {
                    PleUploadMode::Blocking
                };
                stage_ple_rows(
                    gpu,
                    ple_host,
                    ple,
                    &staged,
                    &ple_rows,
                    ple_rows_f16,
                    n,
                    ple_head_count,
                    ple_row_width,
                    mode,
                )
            };
            if ple_split.is_none() {
                stage_ple(gpu)?;
            }

            // ── Retained-body boundary ────────────────────────────────────────
            // Everything above materialises this forward's inputs — token ids,
            // embeddings, HC stream init, and the PLE staging/upload/gather — and
            // stays outside a retained tape, so it runs on both the HIP path and a
            // replay. Everything below is the body the tape covers, and this
            // boundary owns the whole per-forward retained-body lifecycle (arm,
            // route, close, prepare) so any driver — the generation loop, a
            // benchmark carrier — leaves the controller in a defined state.
            //
            // Eligible only for a plain single-token continuation: the wide-hidden
            // capture is the speculative-verify shape and must never enter the tape.
            // Developer profiles never record: their dense FFN's `silu_f32`
            // launch bypasses the Redline recorder (census 82 launched / 80
            // recorded on tiny-dense-dev-gpu, card B) and the per-row GDN conv
            // carries a position kernarg linear AQL cannot rebind.
            let eligible = n == 1
                && self.ep.is_none()
                && !cpu_deferred_experts
                && config.profile.is_flash_next()
                && !self.lane_scope
                && !gpu.prefill_invariant_scope
                && wide_hidden_capture.is_none()
                && self.qsa_tap.is_none()
                && !mesh_run
                && self.cut_layers.is_none()
                && self.layer_capture.is_none()
                && !hipfire_dispatch::pipeline::qsa_projection_hook_installed()
                // A BorrowPrefill tile (a one-row prompt tail continuing a
                // dirty epoch included) never replays or records a decode
                // body: its tape names canonical donor addresses.
                && !self.borrow_runtimes_active;
            // `HIPFIRE_QWEN4_CUT_LAYERS` is the cut-safe eager schedule: refuse
            // every route that would execute the program some other way.
            if cut_layers.is_some() {
                let conflict = if chunk_dump.is_some() {
                    Some("the chunk dump (HIPFIRE_CHUNK_DUMP_DIR)")
                } else if self.qsa_tap.is_some() {
                    Some("the QSA parity tap")
                } else if self.route_trace.is_some() {
                    Some("the routing trace (HIPFIRE_QWEN4_ROUTE_TRACE)")
                } else if n > 1 && hipfire_dispatch::pipeline::moe_capture::enabled() {
                    Some("HIPFIRE_MOE_ROUTE_DUMP / HIPFIRE_MOE_ACT_CAPTURE")
                } else if self.ep.is_some() {
                    Some("expert parallelism")
                } else {
                    None
                };
                if let Some(conflict) = conflict {
                    return Err(invalid(format!(
                        "{CUT_LAYERS_ENV} cannot run with {conflict}"
                    )));
                }
            }
            gpu.replay.set_forward_eligible(eligible);
            // A manual shadow controller states the executor directly: one prepared
            // tape is compared across the exact-kernarg HIP oracle, the retained
            // transport, and ordinary HIP. The shadow arms never record, so they do
            // not run the production lifecycle below.
            let shadow_route = if eligible {
                gpu.replay.shadow_body_route()
            } else {
                None
            };
            if shadow_route.is_none() {
                match retained_body_action(
                    eligible,
                    gpu.replay.state(),
                    specialized_sealed_moe_retained_admission(),
                ) {
                    RetainedBodyAction::Ineligible => {}
                    RetainedBodyAction::Refuse { reason } => {
                        // The route cannot be retained: poison before the body so this
                        // and every later forward runs on HIP.
                        gpu.replay.poison(reason);
                        eprintln!("[redline] qwen4 retained body unavailable: {reason}");
                    }
                    RetainedBodyAction::Arm { diagnostic } => {
                        diagnostic_capture = diagnostic;
                        // The fused GDN rotation's arrival counters are zeroed once,
                        // on first use; materialize them before the window opens so
                        // that memset is not state the tape would have to replay.
                        rdna_compute::tensor_ops::ensure_gdn_pair_counters(gpu)?;
                        gpu.replay
                            .begin_auto_capture_if_armed()
                            .map_err(|reason| invalid(reason))?;
                    }
                    RetainedBodyAction::Route => {}
                }
            }
            let pm4_route = match shadow_route {
                Some(ShadowBodyRoute::Plan) => gpu.replay.prepared_pm4_plan_ready(),
                Some(_) => false,
                None => gpu.replay.should_route_pm4(),
            };
            let aql_route = match shadow_route {
                Some(ShadowBodyRoute::Plan) => gpu.replay.prepared_aql_plan_ready(),
                Some(_) => false,
                None => gpu.replay.should_route_aql(),
            };
            if ple_split.is_some() && (pm4_route || aql_route || shadow_route.is_some()) {
                return Err(invalid("Qwen4 retained body with deferred PLE staging"));
            }
            if pm4_route || aql_route || shadow_route == Some(ShadowBodyRoute::HipOracle) {
                // The replayed tape (retained transport or the recorded-HIP
                // oracle) never runs the HIP body's per-call identity notes:
                // validate the complete captured binding (tables, codec digests,
                // arena generations) here, before any retained submit.
                let moe = tile_moe;
                if let Err(reason) = gpu.replay.validate_route_identities(|key| {
                    moe.iter().flatten()
                        .flat_map(|layer| layer.retained_identities.iter())
                        .find(|(bound, _)| *bound == key)
                        .map(|(_, identity)| identity.as_str())
                }) {
                    gpu.replay.poison(format!("Qwen4 retained route binding changed: {reason}"));
                    return Err(invalid(format!("Qwen4 retained replay refused: {reason}")));
                }
            }
            // A retained replay never runs while the donor tail is borrowed:
            // `share_require_resident` ran at `forward_chunk_inner` entry for
            // every call that is not a borrowed prompt tile, and a borrowed
            // tile (>= 512 rows) is never retained-eligible.
            debug_assert!(
                !(pm4_route || aql_route || shadow_route == Some(ShadowBodyRoute::HipOracle))
                    || donor_resident,
                "a retained replay was submitted while the donor tail is not resident"
            );
            let routed = if pm4_route {
                // Forced non-default retained route: HIP PLE staging (evidence
                // 54ba6c5a6) is outside the PM4 queue's ordering, so drain the
                // staging stream (not the device) before replay.
                let staging_sync = match gpu.active_stream.as_ref() {
                    Some(stream) => gpu.hip.stream_synchronize(stream),
                    None => gpu.hip.stream_synchronize_default(),
                };
                if let Err(error) = staging_sync {
                    gpu.replay
                        .poison(format!("Qwen4 PM4 staging-stream sync failed: {error}"));
                    return Err(invalid(format!(
                        "Qwen4 retained replay refused: staging-stream sync before PM4 replay failed: {error}"
                    )));
                }
                // SAFETY: the boundary above staged every host input the tape's
                // recorded launches read, and every pointer in the tape is owned by
                // this bundle for the plan's lifetime. `replay_pm4_routed` is the
                // plain PM4 replay unless railgun check mode
                // (`HIPFIRE_RAILGUN_CHECK`) is on.
                match unsafe { gpu.replay_pm4_routed(next_position) } {
                    Ok(_) => true,
                    Err(reason) => {
                        gpu.replay.poison(format!("Qwen4 PM4 replay failed: {reason}"));
                        return Err(invalid(format!("Qwen4 retained replay failed: {reason}")));
                    }
                }
            } else if shadow_route == Some(ShadowBodyRoute::HipOracle) {
                // The exact-kernarg oracle: re-execute the recorded HIP launch
                // sequence. This is the arm that proves the tape names the dispatches
                // ordinary HIP issues, not merely that some retained transport runs.
                let launches = gpu.replay.recorded_launches().len();
                match gpu.replay_recorded_hip_prefix_at(launches, next_position) {
                    Ok(_) => true,
                    Err(reason) => {
                        return Err(invalid(format!(
                            "Qwen4 recorded-HIP oracle failed: {reason}"
                        )));
                    }
                }
            } else if aql_route {
                // SAFETY: same contract as the PM4 route above.
                match unsafe { gpu.replay.replay_linear_aql(next_position) } {
                    Ok(_) => true,
                    Err(reason) => {
                        gpu.replay.poison(format!("Qwen4 AQL replay failed: {reason}"));
                        return Err(invalid(format!("Qwen4 retained replay failed: {reason}")));
                    }
                }
            } else {
                false
            };

            if !routed {
                // Parity harness only: a QSA projection hook counts slots from
                // each forward's first QSA step (no-op with no hook installed).
                hipfire_dispatch::pipeline::reset_qsa_projection_slot();
                let mut capture_hooks =
                    self.layer_capture.as_ref().map(|capture| LayerCaptureHooks {
                        targets: capture_at
                            .iter()
                            .map(|&(at, slot)| (&steps[at] as *const Step<'_> as *const (), slot))
                            .collect(),
                        fired: 0,
                        streams: &self.scratch.streams,
                        rows: capture.rows.sub_offset(
                            capture.row * capture_layers.len() * config.hidden_size,
                            n * capture_layers.len() * config.hidden_size,
                        ),
                        n,
                        hc: dims.hc_count,
                        hidden: config.hidden_size,
                        slots: capture_layers.len(),
                        bf16: bf16_state,
                    });
                let mut execute = |gpu: &mut Gpu, steps: &[Step<'_>]| {
                    match capture_hooks.as_mut() {
                        Some(hooks) => {
                            execute_validated_steps_with_moe_hooks(gpu, &ctx, steps, hooks)
                        }
                        None => execute_validated_steps(gpu, &ctx, steps),
                    }
                    .map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!(
                            "execute Qwen4 layer program: {error:?}"
                        ))
                    })
                };
                if !dump_ends.is_empty() && expert_stage.is_none() && self.qsa_tap.is_none() {
                    // `HIPFIRE_CHUNK_DUMP_DIR`: run the chunk in segments that end after
                    // each selected layer and dump that boundary (diagnostic only).
                    let dump = chunk_dump.expect("dump ends imply the dump is enabled");
                    let mut start = 0;
                    for index in 0..steps.len() {
                        if ple_split == Some(index) {
                            execute(gpu, &steps[start..index])?;
                            stage_ple(gpu)?;
                            start = index;
                        }
                        if let Some(&(layer, _, bf16_grouped)) =
                            dump_ends.iter().find(|(_, end, _)| *end == index + 1)
                        {
                            execute(gpu, &steps[start..=index])?;
                            start = index + 1;
                            dump.dump_layer(
                                gpu,
                                &self.scratch,
                                &config,
                                layer,
                                bundle.state.position,
                                n,
                                bf16_grouped,
                            )
                            .map_err(|error| invalid(format!("chunk dump: {error}")))?;
                        }
                    }
                    execute(gpu, &steps[start..])?;
                } else if let Some(tap) = self.qsa_tap.as_mut() {
                    // Parity tap: run up to and including each QSA step, then
                    // let the observer read that layer's scratch and state.
                    let mut start = 0;
                    let mut qsa_slot = 0;
                    for (index, step) in steps.iter().enumerate() {
                        if ple_split == Some(index) {
                            execute(gpu, &steps[start..index])?;
                            stage_ple(gpu)?;
                            start = index;
                        }
                        if let Step::IndexedAttention(op) = step {
                            execute(gpu, &steps[start..=index])?;
                            start = index + 1;
                            tap(gpu, qsa_slot, op).map_err(invalid)?;
                            qsa_slot += 1;
                        }
                    }
                    execute(gpu, &steps[start..])?;
                } else if let Some(stage) = expert_stage {
                    // One interpreter call over the whole program: the stage
                    // waits and refills run as MoE hooks, so the fusions across
                    // a MoE call (HC read/write pairing, the combine zero-init,
                    // the shared-down fold) apply as on an unstaged forward.
                    let mut hooks = StagedMoeHooks {
                        stage,
                        bundle,
                        moe: tile_moe,
                        host_sources: self.ep_reserve.as_ref().map(|reserve| &reserve.host_sources),
                        layer: 0,
                        error: None,
                        capture: capture_hooks
                            .as_mut()
                            .map(|hooks| hooks as &mut dyn MoeStepHooks),
                    };
                    let mut run = |gpu: &mut Gpu, steps: &[Step<'_>]| {
                        execute_validated_steps_with_moe_hooks(gpu, &ctx, steps, &mut hooks)
                            .map_err(|error| {
                                hooks.error.take().unwrap_or_else(|| {
                                    Qwen4GpuForwardError::Dispatch(format!(
                                        "execute Qwen4 layer program: {error:?}"
                                    ))
                                })
                            })
                    };
                    if cut_layers.is_some() {
                        // Cut-safe schedule: separate interpreter runs split at
                        // the PLE step and every recorded cut, one hook state.
                        let (bounds, count) = cut_boundaries(ple_split, &cut_steps[..cut_count]);
                        let mut start = 0usize;
                        for &end in bounds[..count].iter().chain(std::iter::once(&steps.len())) {
                            if end > start {
                                run(gpu, &steps[start..end])?;
                            }
                            if ple_split == Some(end) {
                                stage_ple(gpu)?;
                            }
                            start = end;
                        }
                    } else {
                        match ple_split {
                            Some(split) => {
                                run(gpu, &steps[..split])?;
                                stage_ple(gpu)?;
                                run(gpu, &steps[split..])?;
                            }
                            None => run(gpu, &steps)?,
                        }
                    }
                } else {
                    let trace = self.route_trace.as_ref().filter(|_| n == 1);
                    match (ple_split, trace) {
                        (split, Some(trace)) => {
                            let k = config.num_experts_per_tok;
                            let mut start = 0usize;
                            let mut layer = 0usize;
                            for index in 0..steps.len() {
                                if split == Some(index) {
                                    if start < index {
                                        execute(gpu, &steps[start..index])?;
                                    }
                                    stage_ple(gpu)?;
                                    start = index;
                                }
                                if matches!(steps[index], Step::Moe(_)) {
                                    execute(gpu, &steps[start..=index])?;
                                    start = index + 1;
                                    gpu.copy_d2d(
                                        &self.scratch.moe_topk_indices,
                                        &trace.device.sub_offset(layer * k, k),
                                        k * std::mem::size_of::<i32>(),
                                    )?;
                                    layer += 1;
                                }
                            }
                            if start < steps.len() {
                                execute(gpu, &steps[start..])?;
                            }
                            traced = true;
                        }
                        // Developer capture, off by default (`HIPFIRE_MOE_ROUTE_DUMP`,
                        // `HIPFIRE_MOE_ACT_CAPTURE`): run the steps up to and including each MoE step
                        // eagerly, tagged with its layer ordinal and the chunk's first position, so
                        // the shared sealed-MoE hooks (`hipfire_dispatch::pipeline::moe_capture`)
                        // write the route record (`u32 LE layer, rows, topk, position` then
                        // rows*topk i32, slot = row*topk + j) and the activation rows.
                        (split, None) if n > 1 && hipfire_dispatch::pipeline::moe_capture::enabled() => {
                            let (mut start, mut layer) = (0usize, 0usize);
                            for index in 0..steps.len() {
                                if split == Some(index) {
                                    if start < index { execute(gpu, &steps[start..index])?; }
                                    stage_ple(gpu)?;
                                    start = index;
                                }
                                if matches!(steps[index], Step::Moe(_)) {
                                    hipfire_dispatch::pipeline::moe_capture::set_site(layer, bundle.state.position);
                                    execute(gpu, &steps[start..=index])?;
                                    start = index + 1;
                                    layer += 1;
                                }
                            }
                            if start < steps.len() { execute(gpu, &steps[start..])?; }
                        }
                        // `HIPFIRE_QWEN4_CUT_LAYERS`: separate interpreter runs split
                        // at the PLE step and every recorded cut (fresh fusion state
                        // per segment).
                        (_, None) if cut_layers.is_some() => {
                            let (bounds, count) = cut_boundaries(ple_split, &cut_steps[..cut_count]);
                            let mut start = 0usize;
                            for &end in bounds[..count].iter().chain(std::iter::once(&steps.len())) {
                                if end > start {
                                    execute(gpu, &steps[start..end])?;
                                }
                                if ple_split == Some(end) {
                                    stage_ple(gpu)?;
                                }
                                start = end;
                            }
                        }
                        (Some(split), None) => {
                            execute(gpu, &steps[..split])?;
                            stage_ple(gpu)?;
                            execute(gpu, &steps[split..])?;
                        }
                        (None, None) => execute(gpu, &steps)?,
                    }
                }
                if let Some(hooks) = capture_hooks.as_ref() {
                    if hooks.missed() {
                        return Err(invalid(
                            "DFlash2 layer capture: a capture boundary ran inside a fused step",
                        ));
                    }
                    if let Some(slot) = capture_last {
                        hooks.capture(gpu, slot).map_err(|error| {
                            Qwen4GpuForwardError::Dispatch(format!("{error:?}"))
                        })?;
                    }
                }
            }

            if let Some(capture) = wide_hidden_capture.as_ref().filter(|_| last) {
                let wide_elements = n * wide;
                let source = view(&self.scratch.streams, 0, wide_elements);
                let destination = f32_view(capture, 0, wide_elements);
                if bf16_state {
                    rdna_compute::tensor_ops::hc_state_bf16_to_f32(
                        gpu,
                        &source,
                        &destination,
                        wide_elements,
                    )?;
                } else {
                    gpu.copy_d2d(&source, &destination, source.byte_size())?;
                }
            }

            if let Some(requested_rows) = requested_rows.filter(|_| last) {
                let final_hyper = final_hyper
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 final hyper disappeared before execution"))?;
                let lm_head = lm_head
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 LM head disappeared before execution"))?;
                let logits = logits.ok_or_else(|| {
                    invalid("Qwen4 logits output disappeared before execution")
                })?;
                execute_final_hyper(
                    gpu,
                    dims,
                    final_hyper,
                    &self.scratch.streams,
                    &scratch_desc,
                    n,
                    bf16_state,
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 final hyper: {error:?}"))
                })?;
                let final_hidden = view(&self.scratch.hc_mixed, 0, n * config.hidden_size);
                // The head shares the trunk's basis scratch: each projection
                // consumes its rotated rows before the next one writes.
                let head_rotation =
                    view(&self.scratch.rotation, 0, n * config.hidden_size);
                execute_lm_head(
                    gpu,
                    lm_head,
                    &final_hidden,
                    logits,
                    n,
                    requested_rows,
                    Some(&head_rotation),
                )
                .map_err(
                    |error| {
                        Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 LM head: {error:?}"))
                    },
                )?;

                if let Some(top1) = top1 {
                    argmax_f32(
                        gpu,
                        &ArgmaxF32 { logits,
                        indices: top1,
                        rows: requested_rows,
                        vocab: config.vocab_size, },
                    )?;
                }
            }

            // The restores join the compute stream last, so the head overlaps
            // them and anything not hidden is charged to this forward.
            if let Some(stage) = expert_stage {
                stage.join(gpu)?;
            }

            drop(steps);
            Ok(qsa_commits)
        })();
        // Whatever way the attempt ended, no early id copy outlives the
        // forward: lanes, MTP and the NPU trunk write `token_ids` unfenced.
        let drained = match self.token_readback.as_mut() {
            Some(readback) => readback.complete(&gpu.hip),
            None => Ok(()),
        };
        let attempt = match (attempt, drained) {
            (Ok(_), Err(error)) => Err(error.into()),
            (attempt, _) => attempt,
        };
        let read_tokens: [u32; 1];
        let tokens = match device_read {
            Some(token) => {
                read_tokens = [token];
                &read_tokens[..]
            }
            None => tokens,
        };
        match attempt {
            Ok(qsa_commits) => {
                if let Some(ple) = ple.as_mut() {
                    ple.complete();
                }
                if !last {
                    // Positions, QSA marks and PLE history commit once, with
                    // the last segment.
                    return Ok(());
                }
                for (slot, full_len, raw_len, pooled_len, selected_len, position) in qsa_commits {
                    let state = bundle
                        .state
                        .qsa_mut(slot)
                        .ok_or_else(|| invalid("QSA state slot disappeared"))?;
                    state.full_len = full_len;
                    state.raw_len = raw_len;
                    state.pooled_len = pooled_len;
                    state.selected_len = selected_len;
                    state.position = position;
                    state.clamp_f16_valid();
                }
                let mut next_history = next_history;
                for token in tokens.iter().copied() {
                    next_history.push(token);
                }
                bundle.state.ple_history = next_history;
                // This forward's lease is released, so the reader has a free
                // staging buffer: enqueue the next chunk's rows, hashed from
                // the history just committed, while the GPU runs this chunk.
                // The next forward adopts the materialized ticket when its ids
                // match exactly (bytes are a pure function of the ids), else
                // drops it. Best effort: a refused enqueue only loses overlap.
                if let Some(owner) = bundle.ple.as_ref().filter(|_| !ple_lookahead.is_empty()) {
                    let fit = owner.rows.max_rows_per_prefetch() / ple_head_count;
                    let ahead = &ple_lookahead[..ple_lookahead.len().min(fit)];
                    self.ple_ahead = owner
                        .rows
                        .prefetch(
                            owner.rows.current_epoch(),
                            next_history.row_ids(&owner.metadata, ahead),
                        )
                        .ok();
                }
                bundle.state.commit_row_capture(n);
                bundle.state.position = next_position
                    .checked_add(n)
                    .ok_or_else(|| invalid("Qwen4 forward position overflows at commit"))?;
                // Close the retained window this forward opened, report the census
                // and install (or refuse) the prepared plan. This is the only place
                // that knows both ends of one recorded body, so the window can never
                // outlive the forward that opened it.
                if gpu.replay.should_auto_finalize_capture() {
                    finish_qwen4_capture(gpu, diagnostic_capture);
                }
                if traced {
                    self.write_route_trace(gpu, next_position)?;
                }
                Ok(())
            }
            Err(error) => {
                bundle.state.invalidate_f16_arenas();
                // A staged forward that failed before its join may have left
                // a donor overwritten; nothing may read it until restored.
                let error = match self
                    .expert_stage
                    .as_ref()
                    .map(|stage| stage.restore_blocking(gpu))
                {
                    Some(Err(restore)) => invalid(format!(
                        "{error}; G2 donor expert restore failed: {restore}"
                    )),
                    _ => error,
                };
                if gpu.replay.is_recording() {
                    // A capture window that failed inside the body must not keep
                    // failing every later forward.
                    gpu.replay
                        .poison("qwen4 capture failed; later forwards fall back to HIP");
                    eprintln!("[redline] qwen4 capture failed; later forwards fall back to HIP");
                }
                let aborted = match ple.as_mut() {
                    Some(ple) => ple.abort(),
                    None => Ok(0),
                };
                match aborted {
                    Ok(_) => Err(error),
                    Err(cleanup) => Err(Qwen4GpuForwardError::Ple(format!(
                        "forward attempt failed: {error}; PLE cleanup failed: {cleanup}"
                    ))),
                }
            }
        }
    }
}

/// Close a recorded Qwen4 body: census, then install the prepared plan, or refuse
/// it for a measurement-only capture.
///
/// Prepare failure after a successful HIP warmup poisons the route (sticky
/// fallback) rather than failing the request that already produced output; a
/// window that launched more kernels than it recorded is one such failure.
fn finish_qwen4_capture(gpu: &mut Gpu, diagnostic_capture: bool) {
    let effects = gpu.replay.capture_window_effects();
    let summary = gpu.replay.capture_summary();
    eprintln!(
        "[redline] qwen4 capture census: computed={} recorded={} \
         unique_kernels={} sequence_hash={:016x}",
        effects.launches, summary.launch_count, summary.unique_kernel_count, summary.sequence_hash
    );
    eprintln!(
        "[redline] qwen4 capture census: effects inside window — htod={} dtod={} dtoh={} memset={}",
        effects.htod, effects.dtod, effects.dtoh, effects.memset
    );
    if !effects.replayable() {
        eprintln!(
            "[redline] qwen4 capture census: effect-incomplete — {} device copy(ies), {} readback(s) \
             and {} memset(s) inside the window are state the tape cannot replay; prepare refuses it",
            effects.dtod, effects.dtoh, effects.memset
        );
    }
    if diagnostic_capture {
        let reason =
            "diagnostic specialized-MoE capture only; no plan installed (the model runs on \
                      HIP)";
        gpu.replay.poison(reason);
        eprintln!("[redline] qwen4: {reason}");
        return;
    }
    if let Err(reason) = gpu.hip.device_synchronize() {
        let reason = format!("qwen4 capture sync failed: {reason:?}");
        gpu.replay.poison(reason.clone());
        eprintln!("[redline] qwen4 capture could not be closed or prepared: {reason}");
        return;
    }
    if let Err(reason) = gpu.replay.finish_capture() {
        gpu.replay.poison(reason);
        eprintln!("[redline] qwen4 capture could not be closed or prepared: {reason}");
        return;
    }
    let launches = gpu.replay.recorded_launches().len();
    let prepared = if gpu.replay.uses_pm4_transport() {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, launches)
            .map(|_| ())
    } else {
        gpu.replay
            .prepare_linear_aql(gpu.device_id as usize)
            .map(|_| ())
    };
    match prepared {
        Ok(()) => eprintln!(
            "[redline] qwen4 retained body prepared: transport={} dispatches={launches}",
            gpu.replay.transport_name()
        ),
        Err(reason) => {
            let reason = format!("Redline prepare after warmup failed: {reason}");
            gpu.replay.poison(reason.clone());
            eprintln!("[redline] falling back to HIP: {reason}");
        }
    }
}

#[cfg(test)]
mod wide_verify_guard_tests {
    use super::{qwen4_ar_launch, qwen4_spec_logit_rows, Qwen4ArLaunch};
    use rdna_compute::replay::{ReplayBackendRequest as Request, ReplayState as State};

    #[test]
    fn ar_launch_truth_table() {
        use Qwen4ArLaunch::{Eager, Recorded};
        // Retained backend off, or poisoned: AR is ordinary HIP.
        assert_eq!(
            qwen4_ar_launch(true, Request::Hip, State::Hip, false),
            Eager
        );
        assert_eq!(
            qwen4_ar_launch(true, Request::Auto, State::Fallback, false),
            Eager
        );
        assert_eq!(
            qwen4_ar_launch(true, Request::Auto, State::Hip, false),
            Eager
        );
        // Not a retained candidate (EP, lane scope, tap, non Flash-Next): eager
        // whatever the controller holds.
        for state in [State::Armed, State::RecordingWarmup, State::Ready] {
            assert_eq!(qwen4_ar_launch(false, Request::Auto, state, false), Eager);
        }
        // Every live retained state records or replays AR.
        for state in [
            State::Armed,
            State::RecordingWarmup,
            State::Captured,
            State::ShadowValidated,
            State::Ready,
        ] {
            assert_eq!(qwen4_ar_launch(true, Request::Auto, state, false), Recorded);
        }
        assert_eq!(
            qwen4_ar_launch(true, Request::Shadow, State::Armed, false),
            Recorded
        );
        // A hipGraph capture closes the same routes, even on a non-candidate.
        assert_eq!(
            qwen4_ar_launch(false, Request::Hip, State::Hip, true),
            Recorded
        );
    }

    #[test]
    fn physical_verify_rows_respect_actual_scratch() {
        let physical = hipfire_config::QWEN4_VERIFY_MAX_ROWS;
        assert_eq!(physical, 64);
        for scratch_rows in 0..=physical + 1 {
            assert_eq!(
                qwen4_spec_logit_rows(scratch_rows),
                scratch_rows.min(physical)
            );
        }
        assert_eq!(qwen4_spec_logit_rows(usize::MAX), physical);
    }

    #[test]
    fn scoped_verify_width_never_widens_non_single_fn() {
        use rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS;
        assert_eq!(FEW_ROW_ROUTE_ROWS, 8);
        for rows in 1..=hipfire_config::QWEN4_VERIFY_MAX_ROWS {
            assert_eq!(super::qwen4_scoped_verify_rows(rows, true, None), rows);
            assert_eq!(
                super::qwen4_scoped_verify_rows(rows, false, None),
                rows.min(FEW_ROW_ROUTE_ROWS)
            );
            assert_eq!(
                super::qwen4_scoped_verify_rows(rows, true, Some(4)),
                rows.min(4)
            );
            assert_eq!(
                super::qwen4_scoped_verify_rows(rows, false, Some(4)),
                rows.min(4)
            );
        }
    }

    #[test]
    fn ar_reference_recomputed_after_retained_fallback() {
        use Qwen4ArLaunch::{Eager, Recorded};
        let transitions = [
            (State::Armed, Recorded),
            (State::Ready, Recorded),
            (State::Fallback, Eager),
            (State::Hip, Eager),
            (State::Armed, Recorded),
        ];
        for (state, expected) in transitions {
            assert_eq!(qwen4_ar_launch(true, Request::Auto, state, false), expected);
        }
    }
}

#[cfg(test)]
mod ep_table_tests {
    use super::{
        qwen4_ep_buffer_index, qwen4_expert_table_entries, Qwen4EpPlacement, Qwen4ExpertEntry,
        Qwen4ExpertOrigin, Qwen4ExpertTableLayout, Qwen4GpuForwardError, Qwen4PeerExpertSource,
        QWEN4_EP_GATHER_BUFFERS, QWEN4_EP_GATHER_MAX_ROWS,
    };

    const EXPERTS: usize = 512;
    const GATE_UP: usize = 1_740_800;
    const DOWN: usize = 870_400;

    fn compact(base: usize, gate_up_base: usize, experts: usize) -> Qwen4PeerExpertSource {
        Qwen4PeerExpertSource {
            gate_up_base: gate_up_base + base,
            gate_up_bytes: experts * GATE_UP,
            down_base: 0x4000_0000 + base,
            down_bytes: experts * DOWN,
        }
    }

    fn dummy() -> Qwen4PeerExpertSource {
        Qwen4PeerExpertSource {
            gate_up_base: 0x9000_0000,
            gate_up_bytes: GATE_UP,
            down_base: 0x9800_0000,
            down_bytes: DOWN,
        }
    }

    fn entries(
        layout: &Qwen4ExpertTableLayout<'_>,
        local: Qwen4PeerExpertSource,
    ) -> Result<Vec<Qwen4ExpertEntry>, Qwen4GpuForwardError> {
        qwen4_expert_table_entries(layout, local, EXPERTS, GATE_UP, DOWN)
    }

    fn message(result: Result<Vec<Qwen4ExpertEntry>, Qwen4GpuForwardError>) -> String {
        match result {
            Err(Qwen4GpuForwardError::Invalid(message)) => message,
            other => panic!("expected an Invalid error, got {other:?}"),
        }
    }

    #[test]
    fn gather_buffer_index_alternates_over_the_trunk_and_isolates_the_mtp_head() {
        assert_eq!(QWEN4_EP_GATHER_BUFFERS, 4);
        assert_eq!(QWEN4_EP_GATHER_MAX_ROWS, 8);
        assert_eq!(qwen4_ep_buffer_index(0, 48), 0);
        assert_eq!(qwen4_ep_buffer_index(1, 48), 1);
        assert_eq!(qwen4_ep_buffer_index(2, 48), 0);
        assert_eq!(qwen4_ep_buffer_index(47, 48), 1);
        assert_eq!(qwen4_ep_buffer_index(48, 48), 2);
        assert_eq!(qwen4_ep_buffer_index(49, 48), 3);
        assert_eq!(qwen4_ep_buffer_index(50, 48), 2);
        assert_eq!(qwen4_ep_buffer_index(1001, 48), 3);
    }

    #[test]
    fn single_table_is_the_contiguous_expert_walk() {
        let local = compact(0, 0x1000_0000, EXPERTS);
        let table = entries(&Qwen4ExpertTableLayout::Single, local).unwrap();
        assert_eq!(table.len(), EXPERTS);
        for (expert, entry) in table.iter().enumerate() {
            assert_eq!(entry.origin, Qwen4ExpertOrigin::Local);
            assert_eq!(entry.gate_up_ptr, local.gate_up_base + expert * GATE_UP);
            assert_eq!(entry.down_ptr, local.down_base + expert * DOWN);
        }
    }

    #[test]
    fn single_table_refuses_a_short_tensor() {
        let mut local = compact(0, 0x1000_0000, EXPERTS);
        local.down_bytes -= 1;
        let text = message(entries(&Qwen4ExpertTableLayout::Single, local));
        assert!(
            text.contains("down expert slot 511") && text.contains("exceeds"),
            "{text}"
        );
    }

    /// Every `(ranks, rank)` of the supported EP rank counts.
    fn ep_rank_cases() -> Vec<(usize, usize)> {
        [2usize, 3, 4, 5, 6, 8]
            .into_iter()
            .flat_map(|ranks| (0..ranks).map(move |rank| (ranks, rank)))
            .collect()
    }

    #[test]
    fn tier_table_preserves_arbitrary_owner_addresses_and_local_ids() {
        let local = compact(0, 0, 0); // No aggregate allocation on a tier load.
        for host_zero in [false, true] {
            let gate: Vec<usize> = (0..EXPERTS)
                .map(|eid| {
                    let base = if (eid % 2 == 0) == host_zero {
                        0x8000_0000
                    } else {
                        0x1000_0000
                    };
                    base + (EXPERTS - eid) * GATE_UP
                })
                .collect();
            let down: Vec<usize> = gate.iter().map(|ptr| ptr + 0x1_0000_0000).collect();
            let table = entries(
                &Qwen4ExpertTableLayout::Tier {
                    gate_up: &gate,
                    down: &down,
                },
                local,
            )
            .unwrap();
            for (eid, entry) in table.iter().enumerate() {
                assert_eq!(entry.origin, Qwen4ExpertOrigin::Tier);
                assert_eq!((entry.gate_up_ptr, entry.down_ptr), (gate[eid], down[eid]));
            }
        }
    }

    #[test]
    fn tier_table_refuses_bad_dimensions_and_null_entries() {
        let gate = vec![0x1000; EXPERTS];
        let mut down = vec![0x2000; EXPERTS];
        let local = compact(0, 0, 0);
        assert!(entries(
            &Qwen4ExpertTableLayout::Tier {
                gate_up: &gate[..EXPERTS - 1],
                down: &down
            },
            local
        )
        .is_err());
        down[EXPERTS - 1] = 0;
        assert!(entries(
            &Qwen4ExpertTableLayout::Tier {
                gate_up: &gate,
                down: &down
            },
            local
        )
        .is_err());
    }

    #[test]
    fn tier_extent_checks_each_owner_span_and_absolute_pointer() {
        use super::tier_expert_extent as check;
        // The shared owner and offset may exceed 4 GiB.
        let base = 0x1_0000_0000usize;
        let offset = 0x2_0000_0000u64;
        let bytes = offset as usize + GATE_UP;
        let ptr = base + offset as usize;
        assert!(check(
            "gate",
            Some((base, bytes)),
            offset,
            GATE_UP as u64,
            GATE_UP,
            ptr
        )
        .is_ok());
        assert!(check(
            "gate",
            Some((base, bytes - 1)),
            offset,
            GATE_UP as u64,
            GATE_UP,
            ptr
        )
        .is_err());
        assert!(check(
            "gate",
            Some((base, bytes)),
            offset,
            GATE_UP as u64 - 1,
            GATE_UP,
            ptr
        )
        .is_err());
        assert!(check(
            "gate",
            Some((base, bytes)),
            offset,
            GATE_UP as u64,
            GATE_UP,
            ptr + 256
        )
        .is_err());
        assert!(check("gate", None, offset, GATE_UP as u64, GATE_UP, ptr).is_err());
        assert!(check("gate", Some((base, bytes)), u64::MAX, 1, 1, ptr).is_err());
        assert!(check(
            "gate",
            Some((usize::MAX - 255, 512)),
            0,
            256,
            256,
            usize::MAX - 255
        )
        .is_err());
        assert!(check("gate", Some((base + 1, bytes)), 0, 256, 256, base + 1).is_err());
    }

    #[test]
    fn ep_table_maps_owned_to_compact_local_and_the_rest_to_the_dummy() {
        for (ranks, rank) in ep_rank_cases() {
            let placement = Qwen4EpPlacement { rank, ranks };
            let local = compact(0, 0x1000_0000, placement.local_count(EXPERTS));
            let layout = Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy(),
            };
            let table = entries(&layout, local).unwrap();
            assert_eq!(table.len(), EXPERTS);
            let mut owned = 0;
            for (expert, entry) in table.iter().enumerate() {
                if expert % ranks == rank {
                    owned += 1;
                    assert_eq!(entry.origin, Qwen4ExpertOrigin::Local);
                    assert_eq!(
                        entry.gate_up_ptr,
                        local.gate_up_base + (expert / ranks) * GATE_UP
                    );
                    assert_eq!(entry.down_ptr, local.down_base + (expert / ranks) * DOWN);
                } else {
                    assert_eq!(entry.origin, Qwen4ExpertOrigin::Dummy);
                    assert_eq!(entry.gate_up_ptr, dummy().gate_up_base);
                    assert_eq!(entry.down_ptr, dummy().down_base);
                }
            }
            assert_eq!(owned, placement.local_count(EXPERTS));
        }
    }

    #[test]
    fn ep_table_requires_exactly_the_compact_expert_count_and_one_dummy_expert() {
        let placement = Qwen4EpPlacement { rank: 1, ranks: 4 };
        let layout = Qwen4ExpertTableLayout::Ep {
            placement,
            dummy: dummy(),
        };
        // A single-device tensor (all 512 experts) is not a compact source.
        let text = message(entries(&layout, compact(0, 0x1000_0000, EXPERTS)));
        assert!(text.contains("exactly 128 experts"), "{text}");
        // One expert short.
        let text = message(entries(&layout, compact(0, 0x1000_0000, 127)));
        assert!(text.contains("exactly 128 experts"), "{text}");
        let mut big_dummy = dummy();
        big_dummy.down_bytes += 4;
        let layout = Qwen4ExpertTableLayout::Ep {
            placement,
            dummy: big_dummy,
        };
        let text = message(entries(&layout, compact(0, 0x1000_0000, 128)));
        assert!(text.contains("dummy expert"), "{text}");
    }

    #[test]
    fn ep_placement_accepts_uneven_splits_and_refuses_out_of_range() {
        let local = compact(0, 0x1000_0000, EXPERTS / 4);
        for placement in [
            Qwen4EpPlacement { rank: 4, ranks: 4 },
            Qwen4EpPlacement { rank: 0, ranks: 0 },
            Qwen4EpPlacement {
                rank: 0,
                ranks: EXPERTS + 1,
            },
        ] {
            let layout = Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy(),
            };
            assert!(entries(&layout, local).is_err(), "{placement:?}");
        }
        // 512 over 3 ranks = 171/171/170: each rank's exact count is accepted.
        for (rank, count) in [(0usize, 171usize), (1, 171), (2, 170)] {
            let placement = Qwen4EpPlacement { rank, ranks: 3 };
            let layout = Qwen4ExpertTableLayout::Ep {
                placement,
                dummy: dummy(),
            };
            assert_eq!(layout.local_experts(EXPERTS).unwrap(), count);
            assert_eq!(
                entries(&layout, compact(0, 0x1000_0000, count))
                    .unwrap()
                    .len(),
                EXPERTS
            );
            // The even-split count is no longer the exact count.
            assert!(entries(&layout, compact(0, 0x1000_0000, EXPERTS / 4)).is_err());
        }
    }

    fn ep_runtime(label: usize, host_mapped: bool) -> super::EpLaneRuntimeFacts {
        super::EpLaneRuntimeFacts {
            label,
            host_mapped,
            packed_mq4: true,
            plain: true,
        }
    }

    #[test]
    fn ep_lane_expert_kind_is_static_or_balanced_from_the_selected_runtime() {
        use super::EpLaneExpertKind::{BalancedHostMappedPacked, StaticResidentPacked};
        let kind = super::classify_ep_lane_expert;
        // Resident trunk layer: the static runtime, no balanced live runtime.
        let resident = Some(ep_runtime(3, false));
        assert_eq!(
            kind(3, true, false, resident, resident),
            Some(StaticResidentPacked)
        );
        // Host-mapped trunk layer with its live balanced runtime selected.
        let mapped = Some(ep_runtime(3, true));
        assert_eq!(
            kind(3, true, true, mapped, mapped),
            Some(BalancedHostMappedPacked)
        );
    }

    #[test]
    fn ep_lane_expert_kind_refuses_every_unproven_binding() {
        let kind = super::classify_ep_lane_expert;
        let resident = Some(ep_runtime(3, false));
        let mapped = Some(ep_runtime(3, true));
        // The row gather is not installed.
        assert_eq!(kind(3, false, false, resident, resident), None);
        assert_eq!(kind(3, false, true, mapped, mapped), None);
        // No selected or no static runtime.
        assert_eq!(kind(3, true, false, None, resident), None);
        assert_eq!(kind(3, true, false, resident, None), None);
        // A balanced selection must be a host-mapped layer, and a host-mapped
        // layer without a live balanced runtime is not static-resident.
        assert_eq!(kind(3, true, true, resident, resident), None);
        assert_eq!(kind(3, true, false, mapped, mapped), None);
        // The balanced layer's static runtime must be the host-mapped one.
        assert_eq!(kind(3, true, true, mapped, resident), None);
        // A runtime labelled for another layer (a balanced table indexed by
        // MoE ordinal selects a neighbour's live runtime).
        let other = Some(ep_runtime(4, false));
        assert_eq!(kind(3, true, false, other, other), None);
        assert_eq!(kind(3, true, false, other, resident), None);
        // Not QT44/QT53 for every expert view, or any CPU/tier/slab/FP8/keep-mask state.
        for (packed_mq4, plain) in [(false, true), (true, false), (false, false)] {
            let bad = Some(super::EpLaneRuntimeFacts {
                packed_mq4,
                plain,
                ..ep_runtime(3, false)
            });
            assert_eq!(kind(3, true, false, bad, bad), None);
            assert_eq!(kind(3, true, false, bad, resident), None);
            assert_eq!(kind(3, true, false, resident, bad), None);
        }
    }

    fn peers(ranks: usize) -> Vec<Qwen4PeerExpertSource> {
        (0..ranks)
            .map(|rank| {
                let count = Qwen4EpPlacement { rank, ranks }.local_count(EXPERTS);
                let mut source = compact(0, 0x1000_0000 + rank * 0x1_0000_0000, count);
                source.down_base += rank * 0x1_0000_0000;
                source
            })
            .collect()
    }

    #[test]
    fn full_table_maps_owned_local_and_non_owned_into_the_owner_peer() {
        for (ranks, rank) in ep_rank_cases() {
            let placement = Qwen4EpPlacement { rank, ranks };
            let peers = peers(ranks);
            let layout = Qwen4ExpertTableLayout::Full {
                placement,
                peers: &peers,
            };
            let table = entries(&layout, peers[rank]).unwrap();
            assert_eq!(table.len(), EXPERTS);
            for (expert, entry) in table.iter().enumerate() {
                let owner = expert % ranks;
                let slot = expert / ranks;
                let expected_origin = if owner == rank {
                    Qwen4ExpertOrigin::Local
                } else {
                    Qwen4ExpertOrigin::Peer(owner)
                };
                assert_eq!(entry.origin, expected_origin, "expert {expert}");
                assert_eq!(
                    entry.gate_up_ptr,
                    peers[owner].gate_up_base + slot * GATE_UP,
                    "expert {expert}"
                );
                assert_eq!(
                    entry.down_ptr,
                    peers[owner].down_base + slot * DOWN,
                    "expert {expert}"
                );
            }
        }
    }

    #[test]
    fn full_table_refuses_inconsistent_or_short_peers() {
        let placement = Qwen4EpPlacement { rank: 0, ranks: 2 };
        let good = peers(2);
        let layout = Qwen4ExpertTableLayout::Full {
            placement,
            peers: &good,
        };
        // Peer list for the wrong rank count.
        let wrong_count = &good[..1];
        let layout_count = Qwen4ExpertTableLayout::Full {
            placement,
            peers: wrong_count,
        };
        let text = message(entries(&layout_count, good[0]));
        assert!(text.contains("needs 2 peer sources"), "{text}");
        // `peers[rank]` must be this rank's own tensors.
        let text = message(entries(&layout, good[1]));
        assert!(text.contains("does not name this rank's own"), "{text}");
        // A peer whose tensor is one expert short is caught by the slot bounds.
        let mut short = good.clone();
        short[1].gate_up_bytes -= GATE_UP;
        let layout_short = Qwen4ExpertTableLayout::Full {
            placement,
            peers: &short,
        };
        let text = message(entries(&layout_short, short[0]));
        assert!(text.contains("exactly 256 experts"), "{text}");
        // Uneven peers: rank 1 of 3 holds 171 experts; one short (and one over) is refused.
        let placement = Qwen4EpPlacement { rank: 0, ranks: 3 };
        let uneven = peers(3);
        let layout_uneven = Qwen4ExpertTableLayout::Full {
            placement,
            peers: &uneven,
        };
        assert_eq!(entries(&layout_uneven, uneven[0]).unwrap().len(), EXPERTS);
        for delta in [-1isize, 1] {
            let mut skewed = uneven.clone();
            skewed[2].gate_up_bytes =
                (skewed[2].gate_up_bytes as isize + delta * GATE_UP as isize) as usize;
            let layout_skewed = Qwen4ExpertTableLayout::Full {
                placement,
                peers: &skewed,
            };
            let text = message(entries(&layout_skewed, skewed[0]));
            assert!(text.contains("exactly 170 experts"), "{text}");
        }
        // A null peer base is refused rather than mapped (the two-rank layout).
        let mut null = good.clone();
        null[1].down_base = 0;
        let layout_null = Qwen4ExpertTableLayout::Full {
            placement: Qwen4EpPlacement { rank: 0, ranks: 2 },
            peers: &null,
        };
        let text = message(entries(&layout_null, null[0]));
        assert!(text.contains("null"), "{text}");
    }
}

/// Split-admission `enabled` decision of one layer build. Trunk layers need
/// the load's D2 default (or the explicit `cpu` mode for tier layers) on a
/// tier-capable build; the native MTP head (`head_split`, untiered and never
/// D1) is armed by its own knob and is still refused when a keep mask prunes
/// its experts.
pub(crate) fn d2_split_enabled(
    tier: bool,
    explicit_cpu_split: bool,
    d2_default: bool,
    allow_tiers: bool,
    label_in_trunk: bool,
    kept_none: bool,
    head_split: bool,
) -> bool {
    ((if tier {
        explicit_cpu_split || d2_default
    } else {
        d2_default
    }) && allow_tiers
        && label_in_trunk
        || head_split)
        && kept_none
}

#[cfg(test)]
mod d2_split_enabled_tests {
    use super::d2_split_enabled;

    /// The trunk formula, verbatim from before the head knob existed.
    fn trunk(tier: bool, explicit: bool, d2: bool, allow: bool, label: bool, kept: bool) -> bool {
        (if tier { explicit || d2 } else { d2 }) && allow && label && kept
    }

    #[test]
    fn trunk_admission_is_unchanged_for_every_input_without_the_head_knob() {
        for bits in 0..64u32 {
            let b = |n: u32| bits & (1 << n) != 0;
            assert_eq!(
                d2_split_enabled(b(0), b(1), b(2), b(3), b(4), b(5), false),
                trunk(b(0), b(1), b(2), b(3), b(4), b(5)),
                "{bits:06b}"
            );
        }
    }

    #[test]
    fn head_arms_without_the_trunk_gates_and_a_pruned_head_is_refused() {
        // Today's head: untiered, D1-free, no D2 default, not a trunk label.
        assert!(!d2_split_enabled(
            false, false, false, false, false, true, false
        ));
        assert!(d2_split_enabled(
            false, false, false, false, false, true, true
        ));
        assert!(!d2_split_enabled(
            false, false, false, false, false, false, true
        ));
        // The knob never un-refuses a keep-masked trunk-shaped layer either.
        assert!(!d2_split_enabled(
            false, true, true, true, true, false, true
        ));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_qwen4_cut_layers_accepts_valid_lists() {
        let cuts = super::parse_qwen4_cut_layers("9,17,25,33,41", 48, Some(0)).unwrap();
        assert_eq!(cuts.as_slice(), &[9, 17, 25, 33, 41]);
        let cuts = super::parse_qwen4_cut_layers("2,24,47", 48, Some(1)).unwrap();
        assert_eq!(cuts.as_slice(), &[2, 24, 47]);
        let cuts = super::parse_qwen4_cut_layers(" 24 ", 48, None).unwrap();
        assert_eq!(cuts.as_slice(), &[24]);
        let cuts = super::parse_qwen4_cut_layers("1,2,3,4,5,6,7", 48, None).unwrap();
        assert_eq!(cuts.as_slice(), &[1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn parse_qwen4_cut_layers_rejects_invalid_lists() {
        for (raw, layers, ple) in [
            ("", 48, None),
            ("0", 48, None),
            ("1", 48, Some(1)),
            ("48", 48, None),
            ("9,9", 48, None),
            ("17,9", 48, None),
            ("1,2,3,4,5,6,7,8", 48, None),
            ("9,,17", 48, None),
            ("9,", 48, None),
            ("x", 48, None),
            ("-3", 48, None),
            ("9;17", 48, None),
        ] {
            assert!(
                super::parse_qwen4_cut_layers(raw, layers, ple).is_err(),
                "{raw:?} with {layers} layers and PLE {ple:?} must be rejected"
            );
        }
        let error = super::parse_qwen4_cut_layers("48", 48, None).unwrap_err();
        assert!(error.contains("48"), "{error}");
    }

    #[test]
    fn cut_boundaries_merge_the_ple_split_before_the_cuts() {
        let (bounds, count) = super::cut_boundaries(Some(3), &[10, 20]);
        assert_eq!(&bounds[..count], &[3, 10, 20]);
        let (bounds, count) = super::cut_boundaries(None, &[10, 20]);
        assert_eq!(&bounds[..count], &[10, 20]);
        let (bounds, count) = super::cut_boundaries(Some(3), &[]);
        assert_eq!(&bounds[..count], &[3]);
    }

    #[test]
    fn gfx942_trunk_profile_is_pinned_without_changing_rdna() {
        for knob in [false, true] {
            assert!(super::trunk_prefill_invariant_scope(true, "gfx942", knob));
            assert!(!super::trunk_prefill_invariant_scope(false, "gfx942", knob));
            assert_eq!(
                super::trunk_prefill_invariant_scope(true, "gfx1151", knob),
                knob
            );
            assert!(!super::trunk_prefill_invariant_scope(true, "gfx1201", knob));
        }
        use super::Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
        let unmeasured = || crate::host_link::HostLink::Unmeasured;
        assert_eq!(
            super::qwen4_prefill_chunk_default("gfx942", Full, unmeasured),
            1536
        );
        assert_eq!(
            super::qwen4_prefill_chunk_default("gfx1151", Full, unmeasured),
            8192
        );
        assert_eq!(
            super::qwen4_prefill_chunk_default("gfx1201", Full, unmeasured),
            4096
        );
        assert_eq!(
            super::qwen4_prefill_chunk_default("gfx1201", FnSymGfx1201, unmeasured),
            16384
        );
    }

    #[test]
    fn gfx1201_chunk_follows_the_scratch_mode_and_explicit_env_wins() {
        use super::Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
        use super::{prefill_chunk_requested_from as chunk, qwen4_prefill_chunk_rungs};
        let probe = || unreachable!("probed");
        // The compact phase-arena scratch takes 16384 rows, the Full plan 4096.
        assert_eq!(chunk("gfx1201", None, usize::MAX, Full, probe), 4096);
        assert_eq!(
            chunk("gfx1201", None, usize::MAX, FnSymGfx1201, probe),
            16384
        );
        // An explicit value is authoritative under both plans.
        for mode in [Full, FnSymGfx1201] {
            assert_eq!(
                chunk("gfx1201", Some("8192"), usize::MAX, mode, probe),
                8192
            );
            assert_eq!(
                chunk("gfx1201", Some("2048"), usize::MAX, mode, probe),
                2048
            );
            assert_eq!(
                chunk("gfx1201", Some("16384"), usize::MAX, mode, probe),
                16384
            );
            assert_eq!(chunk("gfx1201", Some("16384"), 4096, mode, probe), 4096);
        }
        // Never above max_seq.
        assert_eq!(chunk("gfx1201", None, 6144, FnSymGfx1201, probe), 6144);
        assert_eq!(chunk("gfx1201", None, 1024, Full, probe), 1024);
        // The scratch mode can only raise a gfx1201 default, never lower it,
        // so a bound taken with `FnSymGfx1201` covers every mode.
        for arch in ["gfx1100", "gfx1151", "gfx942", "gfx1030", "gfx1201"] {
            assert!(
                super::qwen4_prefill_chunk_baseline_default(arch, FnSymGfx1201)
                    >= super::qwen4_prefill_chunk_baseline_default(arch, Full),
                "{arch}"
            );
        }
        // A 16384 request that does not fit tries 8192 first.
        assert_eq!(
            qwen4_prefill_chunk_rungs(16384).collect::<Vec<_>>(),
            [16384, 8192, 4096, 2048, 1536]
        );
    }

    #[test]
    fn pp_prefill_chunk_is_4096_on_gfx1201_and_otherwise_the_single_card_default() {
        use super::Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
        use super::{pp_prefill_chunk_requested_from as pp, prefill_chunk_requested_from};
        use crate::host_link::HostLink::Unmeasured;
        let probe = || unreachable!("probed");
        // Both scratch plans pipeline 4096-row chunks under PP.
        assert_eq!(pp("gfx1201", None, usize::MAX, FnSymGfx1201, probe), 4096);
        assert_eq!(pp("gfx1201", None, usize::MAX, Full, probe), 4096);
        // Explicit env wins, rounded to whole 256-row tiles.
        assert_eq!(
            pp("gfx1201", Some("16384"), usize::MAX, FnSymGfx1201, probe),
            16384
        );
        assert_eq!(pp("gfx1201", Some("5000"), usize::MAX, Full, probe), 4864);
        // Capped at max_seq.
        assert_eq!(pp("gfx1201", None, 2048, Full, probe), 2048);
        assert_eq!(pp("gfx1201", Some("16384"), 8192, Full, probe), 8192);
        // Other arches keep the single-card default.
        for arch in ["gfx1100", "gfx1151"] {
            for mode in [FnSymGfx1201, Full] {
                assert_eq!(
                    pp(arch, None, usize::MAX, mode, || Unmeasured),
                    prefill_chunk_requested_from(arch, None, usize::MAX, mode, || Unmeasured),
                );
            }
        }
    }

    #[test]
    fn gfx1100_chunk_is_8192_only_on_a_measured_narrow_discrete_link() {
        use super::prefill_chunk_requested_from;
        use super::Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
        use crate::host_link::HostLink::{self, Discrete, Unified, Unmeasured};
        let chunk = |arch: &str, link: HostLink| {
            prefill_chunk_requested_from(arch, None, usize::MAX, Full, move || link)
        };
        // Narrow discrete link: the larger admitted rung.
        assert_eq!(chunk("gfx1100", Discrete { gbps: 12.0 }), 8192);
        assert_eq!(chunk("gfx1100", Discrete { gbps: 15.9 }), 8192);
        // Fast, unmeasured and unified links keep today's 1536.
        assert_eq!(chunk("gfx1100", Discrete { gbps: 16.0 }), 1536);
        assert_eq!(chunk("gfx1100", Discrete { gbps: 26.0 }), 1536);
        assert_eq!(chunk("gfx1100", Unmeasured), 1536);
        assert_eq!(chunk("gfx1100", Unified), 1536);
        // No other arch changes: gfx1201 Full stays 4096 on any link (Gen5
        // x16 R9700 included), gfx1151 stays 8192, the rest 1536.
        for link in [
            Discrete { gbps: 4.0 },
            Discrete { gbps: 27.3 },
            Unmeasured,
            Unified,
        ] {
            assert_eq!(chunk("gfx1201", link), 4096, "{link:?}");
            assert_eq!(chunk("gfx1151", link), 8192, "{link:?}");
            assert_eq!(chunk("gfx942", link), 1536, "{link:?}");
            assert_eq!(chunk("gfx1030", link), 1536, "{link:?}");
        }
        // Only gfx1100 evaluates the link (so the other arches never probe),
        // and an explicit chunk request never does.
        for (arch, mode, rows) in [
            ("gfx1201", Full, 4096),
            ("gfx1201", FnSymGfx1201, 16384),
            ("gfx1151", Full, 8192),
            ("gfx942", Full, 1536),
            ("gfx1030", Full, 1536),
        ] {
            assert_eq!(
                prefill_chunk_requested_from(arch, None, usize::MAX, mode, || unreachable!(
                    "probed"
                )),
                rows,
                "{arch}"
            );
        }
        assert_eq!(
            prefill_chunk_requested_from("gfx1100", Some("2048"), usize::MAX, Full, || {
                unreachable!("probed")
            }),
            2048
        );
        // max_seq still caps it, and the attach-time rung search can step down.
        assert_eq!(
            prefill_chunk_requested_from("gfx1100", None, 4096, Full, || Discrete { gbps: 8.0 }),
            4096
        );
        assert_eq!(
            super::qwen4_prefill_chunk_rungs(8192).collect::<Vec<_>>(),
            [8192, 4096, 2048, 1536]
        );
    }

    #[test]
    fn placement_chunk_is_the_requested_chunk_and_only_gfx1100_reads_the_link() {
        use super::prefill_chunk_requested_from;
        use super::Qwen4ForwardScratchMode as Mode;
        use crate::host_link::HostLink::{Discrete, Unified, Unmeasured};
        // gfx1100's chunk does not depend on the scratch mode.
        fn chunk(
            arch: &str,
            env: Option<&str>,
            max_seq: usize,
            link: impl FnOnce() -> crate::host_link::HostLink,
        ) -> usize {
            prefill_chunk_requested_from(arch, env, max_seq, Mode::Full, link)
        }
        // The expert placement reserve is sized from the requested chunk, so
        // a measured narrow discrete gfx1100 link is the one default that
        // places for 8192 rows (ceding resident layers); every other link
        // on gfx1100 keeps its 1536-row placement.
        assert_eq!(
            chunk("gfx1100", None, usize::MAX, || Discrete { gbps: 7.1 }),
            8192
        );
        for link in [
            Discrete { gbps: 16.0 },
            Discrete { gbps: 26.0 },
            Unmeasured,
            Unified,
        ] {
            assert_eq!(
                chunk("gfx1100", None, usize::MAX, || link),
                1536,
                "{link:?}"
            );
        }
        // Every other arch keeps its baseline and never evaluates the link
        // (gfx1201, gfx1151 and the rest never probe).
        for (arch, mode, rows) in [
            ("gfx1151", Mode::Full, 8192),
            ("gfx1201", Mode::Full, 4096),
            ("gfx1201", Mode::FnSymGfx1201, 16384),
            ("gfx942", Mode::Full, 1536),
            ("gfx1030", Mode::Full, 1536),
        ] {
            assert_eq!(
                super::qwen4_prefill_chunk_baseline_default(arch, mode),
                rows
            );
            assert_eq!(
                prefill_chunk_requested_from(arch, None, usize::MAX, mode, || unreachable!(
                    "probed"
                )),
                rows,
                "{arch} {mode:?}"
            );
        }
        // An explicit value is authoritative on every arch and link and never
        // probes, rounded to tiles and capped by max_seq; unparsable or
        // sub-tile values fall back to the default for the link.
        let probe = || unreachable!("probed");
        assert_eq!(chunk("gfx1100", Some("8192"), usize::MAX, probe), 8192);
        assert_eq!(chunk("gfx1100", Some("1536"), usize::MAX, probe), 1536);
        assert_eq!(chunk("gfx1100", Some(" 2100 "), usize::MAX, probe), 2048);
        assert_eq!(chunk("gfx1100", Some("8192"), 4096, probe), 4096);
        assert_eq!(
            chunk("gfx1100", Some("128"), usize::MAX, || Unmeasured),
            1536
        );
        assert_eq!(
            chunk("gfx1100", Some("junk"), usize::MAX, || Unmeasured),
            1536
        );
        assert_eq!(
            chunk("gfx1100", Some("junk"), usize::MAX, || Discrete {
                gbps: 7.1
            }),
            8192
        );
        assert_eq!(
            chunk("gfx1100", None, 1024, || Discrete { gbps: 7.1 }),
            1024
        );
    }

    #[test]
    fn expert_stage_min_rows_follows_the_effective_prefill_chunk() {
        use super::{
            prefill_chunk_requested_from, resolve_expert_stage_min_rows as resolve,
            StageMinRowsSource::{Default, Env},
        };
        use crate::host_link::HostLink::{Discrete, Unmeasured};
        let chunk = |arch: &str, env: Option<&str>| {
            prefill_chunk_requested_from(
                arch,
                env,
                usize::MAX,
                super::Qwen4ForwardScratchMode::Full,
                || Unmeasured,
            )
        };
        // gfx1100 on a fast/unmeasured link: the 1536-row chunk is the
        // threshold, so a full chunk stages.
        assert_eq!(chunk("gfx1100", None), 1536);
        assert_eq!(
            resolve("gfx1100", chunk("gfx1100", None), false, None),
            (1536, Default)
        );
        // gfx1201: chunk 4096 (>= 2048) keeps 2048, even if admission picks 1536.
        assert_eq!(chunk("gfx1201", None), 4096);
        assert_eq!(
            resolve("gfx1201", chunk("gfx1201", None), false, None),
            (2048, Default)
        );
        assert_eq!(resolve("gfx1201", 1536, false, None), (2048, Default));
        // An explicit threshold wins on every arch, chunk and link.
        assert_eq!(resolve("gfx1100", 1536, false, Some("512")), (512, Env));
        assert_eq!(resolve("gfx1100", 8192, true, Some("512")), (512, Env));
        assert_eq!(resolve("gfx1201", 4096, false, Some(" 3072 ")), (3072, Env));
        // Below 2 or unparsable keeps the default.
        assert_eq!(resolve("gfx1100", 1536, false, Some("1")), (1536, Default));
        assert_eq!(resolve("gfx1100", 1536, false, Some("x")), (1536, Default));
        assert_eq!(resolve("gfx1100", 8192, true, Some("1")), (768, Default));
        // A larger chunk request caps at the 2048 ceiling on a fast link.
        assert_eq!(chunk("gfx1100", Some("8192")), 8192);
        assert_eq!(
            resolve("gfx1100", chunk("gfx1100", Some("8192")), false, None),
            (2048, Default)
        );
        // A smaller chunk request lowers it; rounding to 256-row tiles applies.
        assert_eq!(chunk("gfx1100", Some("1100")), 1024);
        assert_eq!(
            resolve("gfx1100", chunk("gfx1100", Some("1100")), false, None),
            (1024, Default)
        );
        // A measured narrow discrete link stages from a fixed 768 rows, never
        // lowered to the chunk: a chunk admitted below 768 never stages.
        let narrow = Discrete { gbps: 12.0 }.is_narrow_discrete();
        assert!(narrow);
        assert_eq!(resolve("gfx1100", 8192, narrow, None), (768, Default));
        assert_eq!(resolve("gfx1100", 1536, narrow, None), (768, Default));
        assert_eq!(resolve("gfx1030", 1536, narrow, None), (768, Default));
        assert_eq!(resolve("gfx1100", 768, narrow, None), (768, Default));
        assert_eq!(resolve("gfx1100", 512, narrow, None), (768, Default));
        assert_eq!(resolve("gfx1100", 256, narrow, None), (768, Default));
        // An explicit threshold still wins over the fixed default.
        assert_eq!(resolve("gfx1100", 512, narrow, Some("256")), (256, Env));
        // gfx1201 keeps 2048 on a narrow link too: its behavior is unchanged.
        assert_eq!(resolve("gfx1201", 4096, narrow, None), (2048, Default));
        assert_eq!(resolve("gfx1201", 8192, narrow, None), (2048, Default));
    }

    #[test]
    fn only_a_forced_stage_threshold_reaches_a_narrow_window() {
        use super::expert_stage_takes_narrow_windows as narrow;
        for arch in ["gfx1100", "gfx1201"] {
            // Default thresholds (min(2048, chunk) or 2048) never stage <= 4 rows.
            assert!(!narrow(arch, true, None), "{arch}");
            assert!(!narrow(arch, true, Some("512")), "{arch}");
            assert!(!narrow(arch, true, Some("5")), "{arch}");
            // Below 2 or unparsable keeps the default threshold.
            assert!(!narrow(arch, true, Some("1")), "{arch}");
            assert!(!narrow(arch, true, Some("x")), "{arch}");
            // A forced threshold at or below the 4-row cap does, when staging is on.
            for rows in ["2", "3", "4"] {
                assert!(narrow(arch, true, Some(rows)), "{arch} {rows}");
                assert!(!narrow(arch, false, Some(rows)), "{arch} {rows} stage off");
            }
        }
    }

    /// The decision `routed_view` applies, over every phase and window: the D2
    /// executor is bound only outside a prefill tile and only for a whole
    /// qualified window — at most the narrow cap on the default route, or a
    /// whole logical 1..64-row window inside an explicit lab-wide scope with
    /// allocated capacity. Wiring only, not numeric proof.
    #[test]
    fn d2_executor_is_admitted_only_outside_prefill_and_for_whole_narrow_windows() {
        use super::d2_executor_admitted as admitted;
        let chunk = 8192;
        // Default scope (unarmed): legacy truth unchanged.
        let default = |has_split: bool, d2_phase: bool, rows: usize, route: usize| {
            admitted(has_split, d2_phase, rows, route, false, 8, 4)
        };
        for has_split in [false, true] {
            // Decode (1 row) and a <=4-row verify, outside a prefill tile.
            for rows in 1..=4 {
                assert!(default(has_split, true, rows, rows), "verify rows={rows}");
                // The same <=4-row window INSIDE a prefill tile (a short prompt
                // whose routing window equals its rows) never takes D2.
                assert_eq!(
                    default(has_split, false, rows, rows),
                    !has_split,
                    "prefill-phase rows={rows}"
                );
                // A 1..=4-row tail of a long prefill: routing window is the chunk.
                assert_eq!(
                    default(has_split, true, rows, chunk),
                    !has_split,
                    "tail rows={rows}"
                );
            }
            // Wide windows never take D2 on the default route, in either phase.
            for rows in [5, 8, 768, chunk] {
                assert_eq!(
                    default(has_split, true, rows, rows),
                    !has_split,
                    "wide rows={rows}"
                );
                assert_eq!(
                    default(has_split, false, rows, rows),
                    !has_split,
                    "wide prefill rows={rows}"
                );
            }
        }
    }

    #[test]
    fn d2_executor_wide_truth_is_1_to_64_wiring_only() {
        // Wiring only — MUST NOT be counted as numeric proof (G1/G2 family
        // qualification still required). Lab-wide scope (`verify_rows = 64`)
        // with allocated 64-row capacity admits every whole logical 1..64-row
        // window outside a prefill tile; tails, prefill-phase calls, default
        // scope and short capacity refuse.
        use super::d2_executor_admitted as admitted;
        // Presence arms BOTH lab widths: wide8 AND wide64 execute the FULL
        // logical R1..64 backend (wide8 only closes the GPU few-row projection
        // routes above 8; it must not cap the CPU at 8). Value alone (armed =
        // false) never qualifies wide, whatever verify_rows reads.
        for rows in 1..=64usize {
            for verify in [8usize, 64] {
                assert!(
                    admitted(true, true, rows, rows, true, verify, 64),
                    "lab verify={verify} rows={rows}"
                );
            }
            assert!(
                !admitted(true, false, rows, rows, true, 64, 64),
                "prefill rows={rows}"
            );
            assert!(
                !admitted(true, true, rows, rows + 1, true, 64, 64),
                "tail rows={rows}"
            );
            assert_eq!(
                admitted(true, true, rows, rows, false, 64, 64),
                rows <= 4,
                "unforced rows={rows}"
            );
            // Short capacity binds: a 4-row allocation admits only 1..=4.
            assert_eq!(
                admitted(true, true, rows, rows, true, 64, 4),
                rows <= 4,
                "short-cap rows={rows}"
            );
        }
        // Structural bounds (capacity 64, armed):
        assert!(!admitted(true, true, 65, 65, true, 64, 64));
        assert!(!admitted(true, true, 0, 0, true, 64, 64));
    }

    /// S1 never binds the D2 executor: it binds `routed_view(false, ..)`, so
    /// `d2_phase = false` and a split is never admitted, whatever the tile
    /// shape. D1 and an explicit D2 stay "every call" for decode (the caps are
    /// unchanged).
    #[test]
    fn s1_never_binds_the_d2_executor() {
        use super::{d2_executor_admitted, host_arith_every_call};
        for rows in [1usize, 2, 3, 4, 5, 64, 768] {
            for route_rows in [rows, 8192] {
                for lab in [false, true] {
                    assert!(
                        !d2_executor_admitted(true, false, rows, route_rows, lab, 8, 64),
                        "rows={rows} route_rows={route_rows} lab={lab}"
                    );
                }
            }
        }
        assert!(host_arith_every_call(true, false, false), "D1 every call");
        assert!(
            host_arith_every_call(false, true, true),
            "explicit D2 every call"
        );
        assert!(
            !host_arith_every_call(false, true, false),
            "automatic D2 per call"
        );
    }

    #[test]
    fn automatic_d2_asks_for_eager_only_on_the_calls_that_execute_it() {
        use super::{host_arith_call, host_arith_every_call};
        // (d1, has_split, split_explicit) as `Qwen4GpuForward` holds them.
        let call = |d1, has_split, explicit, prefill, rows, route| {
            host_arith_call(
                host_arith_every_call(d1, has_split, explicit),
                has_split,
                prefill,
                rows,
                route,
                false,
                8,
                4,
            )
        };
        let chunk = 8192;
        for prefill in [false, true] {
            for explicit in [false, true] {
                assert!(
                    !call(false, false, explicit, prefill, 1, 1),
                    "nothing armed"
                );
            }
        }
        // D1 and explicit-mode D2: every call, as before.
        for (prefill, rows, route) in [
            (false, 1, 1),
            (false, 8, 8),
            (true, 512, chunk),
            (true, 3, chunk),
        ] {
            assert!(
                call(false, true, true, prefill, rows, route),
                "explicit {prefill} {rows}"
            );
            assert!(
                call(true, false, false, prefill, rows, route),
                "D1 {prefill} {rows}"
            );
            assert!(
                call(true, false, true, prefill, rows, route),
                "D1 {prefill} {rows}"
            );
        }
        // Automatic D2: decode and <=4-row verify only.
        for rows in 1..=4 {
            assert!(
                call(false, true, false, false, rows, rows),
                "verify rows={rows}"
            );
            assert!(
                !call(false, true, false, true, rows, rows),
                "short prompt rows={rows}"
            );
            assert!(
                !call(false, true, false, true, rows, chunk),
                "prefill tail rows={rows}"
            );
        }
        for rows in [5, 8, 768, chunk] {
            assert!(
                !call(false, true, false, false, rows, rows),
                "wide verify rows={rows}"
            );
            assert!(
                !call(false, true, false, true, rows, rows),
                "wide prefill rows={rows}"
            );
        }
    }

    /// The load-wide speculative refusal does not arm for automatic D2:
    /// explicit wide forwards still bind its executor-less GPU view. This is
    /// not row-exact admission against CPU-arithmetic singleton decode:
    /// `guarded_wide_verify_rows` also applies `native_mtp_cpu_window_rows`,
    /// including automatic D2, so exact native/copy windows retain the CPU cap.
    #[test]
    fn automatic_d2_keeps_the_unarmed_speculative_width() {
        use super::{host_arith_call, host_arith_every_call};
        use crate::mtp_spec::spec_verify_rows_cap;
        use crate::program::QWEN4_CPU_SPLICE_MAX_ROWS;
        let unarmed = host_arith_every_call(false, false, false);
        let automatic = host_arith_every_call(false, true, false);
        assert!(!unarmed && !automatic);
        assert_eq!(
            spec_verify_rows_cap(automatic),
            spec_verify_rows_cap(unarmed)
        );
        assert_eq!(spec_verify_rows_cap(automatic), usize::MAX);
        for load_wide in [
            host_arith_every_call(false, true, true),
            host_arith_every_call(true, false, false),
            host_arith_every_call(true, true, true),
        ] {
            assert!(load_wide);
            assert_eq!(spec_verify_rows_cap(load_wide), QWEN4_CPU_SPLICE_MAX_ROWS);
        }
        assert_eq!(
            super::cpu_splice::NARROW_VERIFY_MAX_ROWS,
            QWEN4_CPU_SPLICE_MAX_ROWS
        );
        // Explicit wider forwards (not a row-exact admission certificate)
        // run no host arithmetic on the automatic load.
        let widest = crate::gpu_forward_lanes::LANE_MAX_ROWS.max(16);
        for rows in QWEN4_CPU_SPLICE_MAX_ROWS + 1..=widest {
            assert!(
                !host_arith_call(automatic, true, false, rows, rows, false, 8, 4),
                "rows={rows}"
            );
        }
    }

    /// The narrow-link gfx1100 (XTX-class) fixture: arming the D2 split, by the
    /// explicit mode or by the per-load default, changes no prefill decision.
    /// Every input here is the real production function: the chunk default, the
    /// stage threshold, the stage-vs-split narrow-window conflict, the D2 layer
    /// admission and the shared splice-window predicate.
    #[test]
    fn xtx_prefill_phase_is_identical_with_d2_armed_or_not() {
        use super::cpu_splice::{CpuSplitAdmission, SplitOwnerKind, SplitRefusal};
        use crate::host_link::HostLink::Discrete;
        use hipfire_dispatch::cpu_split::{cpu_splice_window, CPU_SPLIT_MAX_ROWS};

        let link = Discrete { gbps: 12.0 };
        let narrow = link.is_narrow_discrete();
        assert!(narrow);
        let chunk = super::qwen4_prefill_chunk_default(
            "gfx1100",
            super::Qwen4ForwardScratchMode::Full,
            || link,
        );
        assert_eq!(chunk, 8192, "the narrow-link 8192-row prefill chunk");
        let (min_rows, _) = super::resolve_expert_stage_min_rows("gfx1100", chunk, narrow, None);
        assert_eq!(min_rows, 768);
        // Stage keeps taking full-chunk windows while D2 only takes <= 4 rows:
        // the two never meet (D2's narrow-window refusal stays off).
        assert!(min_rows > CPU_SPLIT_MAX_ROWS && min_rows <= chunk);
        let narrow_stage = super::expert_stage_takes_narrow_windows("gfx1100", true, None);
        assert!(!narrow_stage);

        for kind in [
            SplitOwnerKind::Tier,
            SplitOwnerKind::WholeLayer,
            SplitOwnerKind::Slab,
        ] {
            for d2 in [false, true] {
                let admission = CpuSplitAdmission {
                    enabled: d2,
                    kind: Some(kind),
                    bound_hotset: kind == SplitOwnerKind::Tier,
                    host_experts: 100,
                    flash_next: true,
                    cpu_executed: true,
                    narrow_stage,
                    ..CpuSplitAdmission::default()
                };
                assert_eq!(admission.armed(), d2, "{kind:?}");
                if !d2 {
                    assert_eq!(admission.refusal(), Some(SplitRefusal::NotRequested));
                }
                let armed = admission.armed();
                // Every window wider than the 4-row cap (a full chunk, a short
                // whole prompt) and every 1..=4-row tail of a long prefill (its
                // routing window is still the canonical chunk) is rejected by
                // `cpu_splice_window`, the predicate `Qwen4MoeLayerRuntime::
                // routed_view` applies when it binds the executor-less view for
                // the D2 prefill seal: armed or not they take the GPU route. The
                // binding itself needs a loaded runtime (GPU), so this pins the
                // predicate it is built on.
                for rows in [chunk, 1536, 768, 5, 8] {
                    assert!(
                        !cpu_splice_window(armed, rows, rows),
                        "{kind:?} d2={d2} rows={rows}"
                    );
                }
                for tail in 1..=CPU_SPLIT_MAX_ROWS {
                    assert!(
                        !cpu_splice_window(armed, tail, chunk),
                        "{kind:?} d2={d2} tail={tail}"
                    );
                }
                // Decode and a <=4-row verify window are the only windows D2 takes.
                for rows in 1..=CPU_SPLIT_MAX_ROWS {
                    assert_eq!(
                        cpu_splice_window(armed, rows, rows),
                        d2,
                        "{kind:?} d2={d2} rows={rows}"
                    );
                }
            }
        }
    }

    #[test]
    fn balanced_owners_are_admitted_on_gfx1201_up_to_four_ranks() {
        for ranks in 2..=4 {
            assert_eq!(
                super::qwen4_ep_balanced_owners_refusal("gfx1201", ranks),
                None
            );
        }
        for ranks in 5..=8 {
            assert_eq!(
                super::qwen4_ep_balanced_owners_refusal("gfx1201", ranks).as_deref(),
                Some("needs at most 4 ranks")
            );
        }
        for arch in ["gfx1100", "gfx1151", "gfx1030", "gfx942", ""] {
            for ranks in [2, 6] {
                assert_eq!(
                    super::qwen4_ep_balanced_owners_refusal(arch, ranks).as_deref(),
                    Some("needs gfx1201"),
                    "{arch:?}"
                );
            }
        }
    }

    use super::{
        compact_arena_groups, owner_pitch_extent, qwen4_forward_device_bytes,
        qwen4_spec_logit_rows, slab_codec_tag_bytes, uploaded_ids_range, CompactScratchTerms,
        Qwen4ForwardScratchMode as Mode, Qwen4GpuForwardScratch as Scratch, ScratchField as F,
        ScratchSlot, QWEN4_COMPACT_FALLBACK_ROWS, QWEN4_PLE_NORMED_KEEP_ROWS,
    };
    use crate::config::{compact_test_config, Qwen4Config};

    fn fn_config() -> Qwen4Config {
        compact_test_config()
    }

    fn scratch_bytes(rows: usize, mode: Mode) -> u64 {
        Scratch::device_bytes(&fn_config(), rows, mode).expect("FN scratch bytes")
    }

    fn plan(rows: usize, mode: Mode) -> Vec<ScratchSlot> {
        Scratch::plan(&fn_config(), rows, mode)
            .expect("FN scratch plan")
            .0
    }

    fn field_bytes(slots: &[ScratchSlot], field: F) -> u64 {
        slots[field as usize].bytes().expect("field bytes") as u64
    }

    fn owned_bytes(slots: &[ScratchSlot]) -> u64 {
        slots
            .iter()
            .filter(|slot| slot.owner.is_none())
            .map(|slot| slot.bytes().expect("slot bytes") as u64)
            .sum()
    }

    #[test]
    fn decode_share_capacity_classes() {
        use super::qwen4_decode_share_classes as classes;
        assert_eq!(classes(16384), vec![1024, 2048, 4096, 8192, 16384]);
        assert_eq!(classes(4096), vec![1024, 2048, 4096]);
        assert_eq!(classes(3000), vec![1024, 2048, 3000]);
        assert_eq!(classes(1024), vec![1024]);
        assert_eq!(classes(600), vec![600]);
        assert!(classes(512).is_empty());
        assert!(classes(1).is_empty());
        assert!(classes(0).is_empty());
        for max_chunk in [600, 1024, 1536, 4096, 16384, 20000] {
            let list = classes(max_chunk);
            assert_eq!(list.last(), Some(&max_chunk));
            assert!(list
                .iter()
                .all(|&c| c > super::QWEN4_DECODE_SHARE_RESIDENT_ROWS && c <= max_chunk));
            assert!(list.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn decode_share_borrowed_backing_holds_every_root_and_view() {
        let config = fn_config();
        let scope = crate::weights::Qwen4BandScope::single(&config);
        for mode in [Mode::FnSymGfx1201, Mode::Full] {
            let mut last_span = 0u64;
            for rows in super::qwen4_decode_share_classes(16384) {
                let (slots, _) = Scratch::plan_scoped(&config, rows, mode, &scope).unwrap();
                let backing = Scratch::borrowed_backing(&config, rows, mode, &scope).unwrap();
                assert_eq!(backing.capacity_rows, rows);
                assert!(!backing.roots.is_empty());
                // Sorted by field, 256-aligned, non-overlapping, TokenIds absent, sizes exact.
                let mut end = 0u64;
                let mut previous: Option<usize> = None;
                for root in &backing.roots {
                    assert_ne!(root.field, F::TokenIds);
                    let index = root.field as usize;
                    assert!(previous.map_or(true, |p| p < index), "{rows}: field order");
                    previous = Some(index);
                    assert_eq!(root.offset % 256, 0);
                    assert!(root.offset >= end, "{rows}: {:?} overlaps", root.field);
                    let slot = slots[index];
                    assert!(slot.owner.is_none());
                    assert_eq!(root.bytes, slot.bytes().unwrap() as u64);
                    assert!(root.bytes > 0);
                    end = root.offset + root.bytes;
                }
                assert_eq!(backing.span_bytes % 256, 0);
                assert!(backing.span_bytes >= end && backing.span_bytes - end < 256);
                // Every owning non-empty slot except TokenIds has a root, nothing else does.
                let expected: Vec<F> = slots
                    .iter()
                    .enumerate()
                    .filter(|(i, s)| {
                        s.owner.is_none() && s.elements > 0 && F::ALL[*i] != F::TokenIds
                    })
                    .map(|(i, _)| F::ALL[i])
                    .collect();
                assert_eq!(
                    backing.roots.iter().map(|r| r.field).collect::<Vec<_>>(),
                    expected
                );
                // Every view slot lies inside its owner's root.
                for (index, slot) in slots.iter().enumerate() {
                    let Some(owner) = slot.owner else { continue };
                    if slot.elements == 0 {
                        continue;
                    }
                    let root = backing
                        .roots
                        .iter()
                        .find(|r| r.field == owner)
                        .unwrap_or_else(|| panic!("{rows}: view {index} owner {owner:?} unbacked"));
                    // `slot.offset` and `pitch` count elements of the OWNER's dtype
                    // (`allocate`); a Raw view (`QsaSelected`) is byte elements over an
                    // F32-covered region.
                    let owner_size = slots[owner as usize].dtype.size() as u64;
                    let view_size = slot.dtype.size() as u64;
                    if slot.dtype != slots[owner as usize].dtype {
                        assert_eq!(slot.dtype, rdna_compute::DType::Raw);
                    }
                    let first = slot.offset as u64 * owner_size;
                    assert!(
                        first + slot.elements as u64 * view_size <= root.bytes,
                        "{rows}: view {:?} overruns {owner:?}",
                        F::ALL[index]
                    );
                    // Pitched views address `rows * pitch` elements from their offset
                    // (`segment_view`).
                    if let Some(pitch) = slot.pitch {
                        assert!(
                            first + rows as u64 * pitch as u64 * owner_size <= root.bytes,
                            "{rows}: pitched view {:?} overruns {owner:?}",
                            F::ALL[index]
                        );
                    }
                }
                assert!(backing.span_bytes >= last_span, "{rows}: span not monotone");
                last_span = backing.span_bytes;
                // The charge helper reads the same plan.
                if rows == 16384 {
                    assert_eq!(
                        super::qwen4_decode_share_borrow_bytes(&config, rows, mode),
                        Some(backing.span_bytes)
                    );
                }
            }
        }
    }

    #[test]
    fn decode_share_dedicated_bytes_undercut_the_full_forward() {
        let config = fn_config();
        for mode in [Mode::FnSymGfx1201, Mode::Full] {
            for max_chunk in [1024usize, 4096, 16384] {
                let forward = qwen4_forward_device_bytes(&config, max_chunk, mode).unwrap();
                let dedicated =
                    super::qwen4_decode_share_dedicated_bytes(&config, max_chunk, mode).unwrap();
                assert!(
                    dedicated < forward,
                    "{mode:?} {max_chunk}: {dedicated} >= {forward}"
                );
                let resident = scratch_bytes(super::QWEN4_DECODE_SHARE_RESIDENT_ROWS, mode);
                let [logits, argmax] = super::qwen4_spec_buffer_bytes(&config, max_chunk).unwrap();
                let ids = plan(max_chunk, mode)[F::TokenIds as usize].bytes().unwrap() as u64;
                assert_eq!(dedicated, resident + ids + logits + argmax);
                // The borrowed bank plus the TokenIds owner is the rest of the forward
                // (the resident bank is a strict subset at 512 rows).
                let backing = Scratch::borrowed_backing(
                    &config,
                    max_chunk,
                    mode,
                    &crate::weights::Qwen4BandScope::single(&config),
                )
                .unwrap();
                assert!(backing.span_bytes + ids >= scratch_bytes(max_chunk, mode));
            }
        }
    }

    use rdna_compute::GpuTensor;

    /// A tensor over fake device memory: tests only build and compare pointers
    /// and metadata, nothing is dereferenced or freed.
    fn fake_tensor(address: usize, elements: usize, dtype: rdna_compute::DType) -> GpuTensor {
        GpuTensor {
            // SAFETY: a `Borrowed` buffer over an address no test dereferences.
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    address as *mut std::ffi::c_void,
                    elements * dtype.size(),
                )
            },
            shape: vec![elements],
            dtype,
        }
    }

    #[test]
    fn decode_share_class_selection_table() {
        use super::decode_share_class_for_rows as class_of;
        let classes = super::qwen4_decode_share_classes(16384);
        let table = [
            (1usize, None),
            (300, None),
            (511, None),
            (512, Some(0)),
            (1000, Some(0)),
            (1024, Some(0)),
            (1025, Some(1)),
            (2048, Some(1)),
            (2049, Some(2)),
            (4096, Some(2)),
            (4097, Some(3)),
            (8192, Some(3)),
            (8193, Some(4)),
            (16384, Some(4)),
            (16385, None),
        ];
        for (rows, expected) in table {
            assert_eq!(class_of(&classes, rows), expected, "{rows} rows");
        }
        let odd = super::qwen4_decode_share_classes(3000);
        assert_eq!(odd, vec![1024, 2048, 3000]);
        assert_eq!(class_of(&odd, 2049), Some(2));
        assert_eq!(class_of(&odd, 3001), None);
        assert_eq!(class_of(&[], 4096), None);
    }

    #[test]
    fn decode_share_refusal_names_each_excluded_route() {
        use super::decode_share_refusal as refuse;
        let fn_mode = Mode::FnSymGfx1201;
        assert_eq!(
            refuse("gfx1201", fn_mode, 16384, true, false, false, false),
            None
        );
        assert!(refuse("gfx1201", fn_mode, 16384, false, false, false, false).is_some());
        assert!(refuse("gfx1201", fn_mode, 16384, true, false, true, false).is_some());
        assert!(refuse("gfx1201", Mode::Full, 16384, true, false, false, false).is_some());
        assert!(refuse("gfx1100", fn_mode, 16384, true, false, false, false).is_some());
        assert!(refuse("gfx1201", fn_mode, 512, true, false, false, false).is_some());
        assert!(refuse("gfx1201", fn_mode, 513, true, false, false, false).is_none());
        assert!(refuse("gfx1201", fn_mode, 16384, true, true, false, false).is_some());
        assert!(refuse("gfx1201", fn_mode, 16384, true, false, false, true).is_some());
    }

    #[test]
    fn decode_share_prefill_tile_predicate_table() {
        use super::decode_share_prefill_tile as tile;
        let resident = super::QWEN4_DECODE_SHARE_RESIDENT_ROWS;
        // (rows, argmax forward, lane step, mesh segment) -> prompt tile.
        assert!(tile(resident, false, false, false));
        assert!(tile(16384, false, false, false));
        assert!(!tile(resident - 1, false, false, false));
        assert!(!tile(1, false, false, false));
        // A verify window / decode never reaches the resident rows; the
        // device-argmax decode forward, lane steps and mesh segments never are.
        assert!(!tile(4096, true, false, false));
        assert!(!tile(4096, false, true, false));
        assert!(!tile(4096, false, false, true));
        assert!(resident > hipfire_config::QWEN4_VERIFY_MAX_ROWS);
        assert!(!tile(
            hipfire_config::QWEN4_VERIFY_MAX_ROWS,
            false,
            false,
            false
        ));
    }

    #[test]
    fn decode_share_dirty_borrow_continues_only_inside_the_entry_that_borrowed() {
        use super::{decode_share_borrows_tile as borrows, decode_share_continues_epoch as epoch};
        let resident = super::QWEN4_DECODE_SHARE_RESIDENT_ROWS;
        // Epoch rule (prompt_depth, prompt_borrowed, loan dirty).
        assert!(epoch(1, true, true));
        assert!(epoch(2, true, true));
        assert!(!epoch(0, true, true));
        assert!(!epoch(1, false, true));
        assert!(!epoch(1, true, false));
        assert!(!epoch(0, false, false));
        // (rows, argmax, lane, mesh, continues) -> borrows a prefill bank.
        // Same-entry tail below the resident rows continues the borrow (S6b).
        for rows in [1, 4, resident - 1] {
            assert!(borrows(rows, false, false, false, epoch(1, true, true)));
        }
        // Fresh entry after a finished prompt (dirty loan, nothing borrowed
        // yet): an MTP verify block, draft forward or one-row capture does not
        // borrow, so `share_require_resident` restores first.
        for rows in [1, 2, 4, 17, resident - 1] {
            assert!(!borrows(rows, false, false, false, epoch(1, false, true)));
            assert!(!borrows(rows, false, false, false, epoch(0, false, true)));
        }
        // A fresh entry's tile of at least the resident rows borrows (back to
        // back prefills share the dirty borrow), with or without a dirty loan.
        for dirty in [false, true] {
            assert!(borrows(
                resident,
                false,
                false,
                false,
                epoch(1, false, dirty)
            ));
            assert!(borrows(16384, false, false, false, epoch(1, false, dirty)));
        }
        // A clean loan never continues, whatever the entry state.
        for rows in [1, resident - 1] {
            assert!(!borrows(rows, false, false, false, epoch(1, true, false)));
        }
        // A device-argmax decode forward, a lane step and a mesh segment are
        // never prefill tiles, not even in an entry that borrowed.
        for rows in [1, resident - 1, resident, 4096] {
            assert!(!borrows(rows, true, false, false, epoch(1, true, true)));
            assert!(!borrows(rows, false, true, false, epoch(1, true, true)));
            assert!(!borrows(rows, false, false, true, epoch(1, true, true)));
        }
        // An empty call never continues.
        assert!(!borrows(0, false, false, false, epoch(1, true, true)));
    }

    #[test]
    fn decode_share_tile_moe_selects_the_borrow_twins_only_while_active() {
        use super::tile_moe_of;
        let resident = [Some(1u8), None, Some(3)];
        let classes = [
            vec![Some(11u8), None, Some(13)],
            vec![Some(21u8), None, Some(23)],
        ];
        assert!(std::ptr::eq(
            tile_moe_of(false, 1, &resident, &classes),
            &resident[..]
        ));
        assert!(std::ptr::eq(
            tile_moe_of(true, 0, &resident, &classes),
            &classes[0][..]
        ));
        assert!(std::ptr::eq(
            tile_moe_of(true, 1, &resident, &classes),
            &classes[1][..]
        ));
        // No borrow twins: an inactive selector can only name the resident set.
        let none: [Vec<Option<u8>>; 0] = [];
        assert!(std::ptr::eq(
            tile_moe_of(false, 0, &resident, &none),
            &resident[..]
        ));
    }

    #[test]
    fn decode_share_borrow_view_class_is_the_epochs_dirty_class() {
        use super::borrow_view_class;
        let spans = [458u64 << 20, 632 << 20, 1200 << 20, 2400 << 20, 4700 << 20];
        // Without a loan (`bind`): the tile's own class, clamped to the views.
        for class in 0..5 {
            assert_eq!(borrow_view_class(5, class, &spans, None), class);
        }
        assert_eq!(borrow_view_class(5, 9, &spans, None), 4);
        // With a loan the dirty extent (largest span of the epoch) wins over a
        // smaller tile, and equals the tile's class for the epoch's first tile.
        assert_eq!(borrow_view_class(5, 0, &spans, Some(spans[0])), 0);
        assert_eq!(borrow_view_class(5, 0, &spans, Some(spans[4])), 4);
        assert_eq!(borrow_view_class(5, 2, &spans, Some(spans[1])), 2);
        assert_eq!(borrow_view_class(5, 1, &spans, Some(spans[3])), 3);
        // A single all-donors view is always read.
        assert_eq!(borrow_view_class(1, 3, &spans, Some(spans[4])), 0);
        assert_eq!(borrow_view_class(0, 0, &spans, None), 0);
    }

    /// The reference census: what `allocate` would hand out, built with the same
    /// `bind_views` over fake owners (each owner at its own address).
    fn reference_tensors(slots: &[ScratchSlot]) -> Vec<Option<GpuTensor>> {
        let mut tensors: Vec<Option<GpuTensor>> = slots.iter().map(|_| None).collect();
        for (index, slot) in slots.iter().enumerate() {
            if slot.owner.is_none() && slot.elements != 0 {
                tensors[index] = Some(fake_tensor(
                    0x1_0000_0000 * (index + 1),
                    slot.elements,
                    slot.dtype,
                ));
            }
        }
        Scratch::bind_views(slots, &mut tensors);
        tensors
    }

    #[test]
    fn decode_share_borrowed_bank_matches_the_allocated_census() {
        let config = fn_config();
        let scope = crate::weights::Qwen4BandScope::single(&config);
        let base = 0x7000_0000_0000usize;
        for mode in [Mode::FnSymGfx1201, Mode::Full] {
            for rows in [1024usize, 4096, 16384] {
                let (slots, ple_bytes) = Scratch::plan_scoped(&config, rows, mode, &scope).unwrap();
                let backing = Scratch::borrowed_backing(&config, rows, mode, &scope).unwrap();
                let ids_slot = slots[F::TokenIds as usize];
                let ids_address = 0x5000_0000usize;
                let token_ids = fake_tensor(ids_address, 16384 * 4, ids_slot.dtype);
                let span = backing.span_bytes as usize;
                // A span one byte short is refused; an unaligned base is refused.
                assert!(Scratch::new_borrowed(
                    &config,
                    rows,
                    mode,
                    &scope,
                    base,
                    span - 1,
                    &token_ids
                )
                .is_err());
                assert!(Scratch::new_borrowed(
                    &config,
                    rows,
                    mode,
                    &scope,
                    base + 8,
                    span,
                    &token_ids
                )
                .is_err());
                let (bank, host_ids, host_ple) =
                    Scratch::new_borrowed(&config, rows, mode, &scope, base, span, &token_ids)
                        .unwrap();
                assert_eq!(bank.max_chunk, rows);
                assert_eq!(bank.mode, mode);
                assert_eq!(bank.slots, slots);
                assert_eq!(host_ids.len(), rows * 4);
                assert_eq!(host_ple.len(), ple_bytes);
                assert_eq!(bank.backing, super::ScratchBackingKind::Borrowed);
                let reference = reference_tensors(&slots);
                let owner_of = |field: F| slots[field as usize].owner.unwrap_or(field);
                for (index, slot) in slots.iter().enumerate() {
                    let field = F::ALL[index];
                    let tensor = bank.field_tensor(field);
                    if field == F::TokenIds {
                        assert_eq!(tensor.buf.as_ptr() as usize, ids_address);
                        assert_eq!(tensor.numel(), slot.elements);
                        assert_eq!(tensor.dtype, slot.dtype);
                        continue;
                    }
                    let expected = reference[index].as_ref().unwrap();
                    assert_eq!(
                        tensor.numel(),
                        expected.numel(),
                        "{rows} {field:?} elements"
                    );
                    assert_eq!(tensor.shape, expected.shape, "{rows} {field:?} shape");
                    assert_eq!(tensor.dtype, expected.dtype, "{rows} {field:?} dtype");
                    assert_eq!(
                        tensor.buf.size(),
                        expected.buf.size(),
                        "{rows} {field:?} bytes"
                    );
                    if slot.elements == 0 {
                        continue;
                    }
                    // Offset from the owning root is the allocated bank's.
                    let owner = owner_of(field);
                    let got = tensor.buf.as_ptr() as usize
                        - bank.field_tensor(owner).buf.as_ptr() as usize;
                    let want = expected.buf.as_ptr() as usize
                        - reference[owner as usize].as_ref().unwrap().buf.as_ptr() as usize;
                    assert_eq!(got, want, "{rows} {field:?} offset from {owner:?}");
                    assert_eq!(slot.pitch, bank.slots[index].pitch);
                }
                // Owning roots sit at base + root.offset.
                for root in &backing.roots {
                    let tensor = bank.field_tensor(root.field);
                    assert_eq!(tensor.buf.as_ptr() as usize, base + root.offset as usize);
                    assert_eq!(tensor.buf.size() as u64, root.bytes);
                    assert!(tensor.buf.is_borrowed());
                }
                // Nothing of a borrowed bank is ever freed.
                assert!(super::scratch_free_list(&bank.slots, None, bank.backing).is_empty());
            }
        }
    }

    #[test]
    fn decode_share_free_lists_free_each_owner_once() {
        use super::{scratch_free_list, ScratchBackingKind as K};
        for mode in [Mode::FnSymGfx1201, Mode::Full] {
            let slots = plan(1024, mode);
            let owned = scratch_free_list(&slots, None, K::Owned);
            let expected: Vec<F> = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.owner.is_none() && slot.elements != 0)
                .map(|(index, _)| F::ALL[index])
                .collect();
            assert_eq!(owned, expected);
            assert!(owned.contains(&F::TokenIds));
            // The resident bank's ids belong to the persistent owner.
            let resident = scratch_free_list(&slots, None, K::ExternalIds);
            assert!(!resident.contains(&F::TokenIds));
            let mut rejoined = resident.clone();
            rejoined.insert(0, F::TokenIds);
            assert_eq!(rejoined, owned);
            assert!(scratch_free_list(&slots, None, K::Borrowed).is_empty());
            // No field is freed twice.
            let mut unique = owned.clone();
            unique.dedup();
            assert_eq!(unique, owned);
        }
    }

    /// Resident + prefill banks over fake memory: the swap protocol, dirty
    /// tracking and the prompt capacity accessor.
    #[test]
    fn decode_share_bank_swaps_track_capacity_and_dirty_prefix() {
        let config = fn_config();
        let scope = crate::weights::Qwen4BandScope::single(&config);
        let mode = Mode::FnSymGfx1201;
        let max_chunk = 4096usize;
        let token_ids = fake_tensor(0x5000_0000, max_chunk * 4, rdna_compute::DType::Raw);
        let base = 0x7000_0000_0000usize;
        let classes = super::qwen4_decode_share_classes(max_chunk);
        assert_eq!(classes, vec![1024, 2048, 4096]);
        let spans: Vec<u64> = classes
            .iter()
            .map(|&rows| {
                Scratch::borrowed_backing(&config, rows, mode, &scope)
                    .unwrap()
                    .span_bytes
            })
            .collect();
        let span = *spans.iter().max().unwrap() as usize;
        let bank = |rows: usize, at: usize| {
            Scratch::new_borrowed(&config, rows, mode, &scope, at, span, &token_ids)
                .unwrap()
                .0
        };
        // A fake resident bank (512 rows) at another base; the real one is allocated.
        let mut scratch = bank(super::QWEN4_DECODE_SHARE_RESIDENT_ROWS, 0x6000_0000_0000);
        let mut share = super::DecodeShareBanks {
            max_chunk,
            classes: classes.clone(),
            class_spans: spans.clone(),
            banks: classes.iter().map(|&rows| Some(bank(rows, base))).collect(),
            parked: None,
            active: None,
            token_ids: fake_tensor(0x5000_0000, max_chunk * 4, rdna_compute::DType::Raw),
            backing: super::DecodeShareBacking::Owned(fake_tensor(
                base,
                span,
                rdna_compute::DType::Raw,
            )),
            dirty_bytes: 0,
            stats: Default::default(),
            loan: None,
            prompt_depth: 0,
            prompt_borrowed: false,
        };
        let prompt_rows = |share: &super::DecodeShareBanks, scratch: &Scratch| {
            super::prompt_chunk_rows_of(Some(share), scratch)
        };
        // Capacity is the prompt chunk whichever bank is selected.
        assert_eq!(scratch.max_chunk, 512);
        assert_eq!(prompt_rows(&share, &scratch), max_chunk);
        assert_eq!(super::prompt_chunk_rows_of(None, &scratch), 512);
        assert!(scratch.max_chunk < share.max_chunk);

        share.swap_in(&mut scratch, 1);
        assert_eq!(scratch.max_chunk, 2048);
        assert_eq!(share.parked.as_ref().map(|bank| bank.max_chunk), Some(512));
        assert_eq!(share.active, Some(1));
        assert!(share.banks[1].is_none());
        assert_eq!(share.dirty_bytes, spans[1]);
        assert_eq!(prompt_rows(&share, &scratch), max_chunk);
        share.swap_out(&mut scratch);
        assert_eq!(scratch.max_chunk, 512);
        assert!(share.parked.is_none() && share.active.is_none());
        assert_eq!(
            share.banks[1].as_ref().map(|bank| bank.max_chunk),
            Some(2048)
        );
        // A smaller class later never lowers the dirty prefix.
        share.swap_in(&mut scratch, 0);
        share.swap_out(&mut scratch);
        assert_eq!(share.dirty_bytes, spans[1]);
        share.swap_in(&mut scratch, 2);
        share.swap_out(&mut scratch);
        assert_eq!(share.dirty_bytes, spans[2]);
        assert_eq!(share.stats.tiles_borrowed, 3);
        assert_eq!(share.stats.bank_swaps, 3);
        // `swap_out` with nothing active is a no-op.
        share.swap_out(&mut scratch);
        assert_eq!(scratch.max_chunk, 512);
    }

    /// `G(r)`: tile-padded grouped rows of `r` rows, top-10 of 512 experts.
    fn grouped_rows(rows: u64) -> u64 {
        let slots = 10 * rows;
        (slots + 15 * slots.min(512)).div_ceil(16) * 16
    }

    #[test]
    fn npu_bank_sharing_covers_every_field_and_never_crosses_a_bank_switch() {
        use super::NpuBankLifetime as L;
        // Event order from pair_scoped, for two layers with PLE on layer 1:
        // initialize A/B, prepare(0,A/B), G(0,A), prepare(1,A), G(0,B),
        // prepare(1,B), G(1,A/B), finish A/B. Each event is a complete
        // same-stream sequence; an interval is one produced value's lifetime,
        // not the lifetime of its allocation.
        let banks = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1];
        let local_init = [(0, 0), (1, 1)];
        let residual = [(0, 10), (1, 11)];
        let input = [(2, 4), (3, 6), (5, 8), (7, 9)];
        // The attention write's gates: prepare's paired read, or the MoE row
        // fold of the bank's previous fragment, to the next fragment.
        let gates = [(2, 4), (3, 6), (4, 8), (5, 8), (6, 9), (7, 9)];
        let ple_rows = [(0, 5), (1, 7)];
        let prepare_and_fragment = [
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (9, 9),
            (10, 10),
            (11, 11),
        ];
        let fragments = [(4, 4), (6, 6), (8, 8), (9, 9)];
        let finish = [(10, 10), (11, 11)];
        // This independent data-flow table is also the full private/shared
        // inventory. HC gates cross from prepare (or the previous fragment's
        // row fold) to the bank's next fragment; fragment-local HC fusions
        // consume their gates before returning. PLE normed rows are retained
        // only for verify rollback, not prompt pairs; session capture copies
        // fixed state, not this field.
        let rules: &[(L, &[F], &[(usize, usize)])] = &[
            (
                L::Initialize,
                &[F::TokenIds, F::EmbeddingRot, F::Embeddings, F::PleStaged],
                &local_init,
            ),
            (L::Residual, &[F::Streams], &residual),
            (L::PreparedInput, &[F::HcMixed], &input),
            (L::PreparedInput, &[F::HcGates], &gates),
            (L::StagedRows, &[F::PleRows], &ple_rows),
            (
                L::PrepareAndFragment,
                &[
                    F::HcNormalized,
                    F::HcLow,
                    F::HcUp,
                    F::Rotation,
                    F::GdnBf16,
                    F::PleKey,
                    F::PleValue,
                    F::PleQuery,
                    F::PleGated,
                    F::PleNormed,
                    F::PleNormedWide,
                    F::PleOutput,
                ],
                &prepare_and_fragment,
            ),
            (
                L::Fragment,
                &[
                    F::Projection,
                    F::Projection2,
                    F::GdnA,
                    F::GdnB,
                    F::GdnGate,
                    F::GdnBeta,
                    F::GdnRecurrentOutput,
                    F::GdnZ,
                    F::GdnOutput,
                    F::QsaIndex,
                    F::QsaQgate,
                    F::QsaK,
                    F::QsaV,
                    F::QsaOutput,
                    F::QsaSelected,
                    F::AttentionOutput,
                    F::RouterLogits,
                    F::MoeXRot,
                    F::MoeGateUp,
                    F::MoeGate,
                    F::MoeUp,
                    F::MoeHidden,
                    F::FfnOutput,
                    F::MoeSharedOutput,
                    F::MoeGateBatch,
                    F::MoeUpBatch,
                    F::MoeRotBatch,
                    F::MoeTopkIndices,
                    F::MoeTopkWeights,
                    F::MoeDownExpanded,
                    F::MoeScalar,
                    F::MoeExpertTokenCounts,
                    F::MoeExpertOffsets,
                    F::MoeSortedSlotIndex,
                    F::MoeExpertTileIds,
                    F::MoeInversePerm,
                    F::MoeYGateUpGrouped,
                    F::MoeYDownGrouped,
                ],
                &fragments,
            ),
            (L::Finish, &[F::Logits], &finish),
        ];
        let mut seen = [false; F::COUNT];
        for &(lifetime, fields, intervals) in rules {
            let crosses_bank = intervals
                .iter()
                .any(|&(first, last)| banks[first..=last].iter().any(|&bank| bank != banks[first]));
            for &field in fields {
                assert!(
                    !std::mem::replace(&mut seen[field as usize], true),
                    "{field:?} repeated"
                );
                assert_eq!(field.npu_bank_lifetime(), lifetime, "{field:?}");
                assert_eq!(
                    field.npu_bank_lifetime().private(),
                    crosses_bank,
                    "{field:?}"
                );
            }
        }
        assert!(seen.into_iter().all(|field| field));
        for (index, field) in F::ALL.into_iter().enumerate() {
            assert_eq!(field as usize, index);
        }
        assert_eq!(
            F::ALL
                .into_iter()
                .filter(|field| field.npu_bank_lifetime().private())
                .collect::<Vec<_>>(),
            [F::Streams, F::HcMixed, F::HcGates, F::PleRows]
        );
    }

    #[test]
    fn npu_private_bank_bytes_are_pinned_at_full_and_partial_rows() {
        let config = fn_config();
        for rows in [512, 8151, 8192] {
            let private = Scratch::npu_bank_device_bytes(&config, rows).unwrap();
            let slots = plan(rows, Mode::Full);
            let expected = [F::Streams, F::HcMixed, F::HcGates, F::PleRows]
                .into_iter()
                .map(|field| field_bytes(&slots, field))
                .sum::<u64>();
            assert_eq!(private, expected);
            assert_eq!(private, rows as u64 * (4 * 2560 + 2560 + 4 + 2560) * 4);
        }
        let private = Scratch::npu_bank_device_bytes(&config, 8192).unwrap();
        assert_eq!(private, 503_447_552);
        assert_eq!(scratch_bytes(8192, Mode::Full) - private, 8_454_551_428);
    }

    /// The kept normed rows cover the widest post-forward reader: a lane step
    /// of 64 lanes of 8 verify rows (a singleton verify window is at most 64).
    #[test]
    fn ple_normed_keep_rows_cover_the_widest_post_forward_reader() {
        assert_eq!(QWEN4_PLE_NORMED_KEEP_ROWS, 512);
        assert!(QWEN4_PLE_NORMED_KEEP_ROWS >= hipfire_config::QWEN4_VERIFY_MAX_ROWS);
        assert!(
            QWEN4_PLE_NORMED_KEEP_ROWS
                >= crate::lane::MAX_LANES * crate::gpu_forward_lanes::LANE_MAX_ROWS
        );
    }

    #[test]
    fn fn_scratch_bytes_are_pinned() {
        assert_eq!(scratch_bytes(4096, Mode::Full), 4_538_503_044);
        assert_eq!(scratch_bytes(8192, Mode::Full), 8_957_998_980);
        assert_eq!(scratch_bytes(256, Mode::FnSymGfx1201), 257_203_716);
        assert_eq!(scratch_bytes(512, Mode::FnSymGfx1201), 395_400_324);
        assert_eq!(scratch_bytes(4096, Mode::FnSymGfx1201), 1_268_469_636);
        assert_eq!(scratch_bytes(8192, Mode::FnSymGfx1201), 2_514_925_444);
        assert_eq!(scratch_bytes(16384, Mode::FnSymGfx1201), 5_007_837_060);
        assert_eq!(
            scratch_bytes(4096, Mode::Full) - scratch_bytes(4096, Mode::FnSymGfx1201),
            3_270_033_408
        );
        assert_eq!(
            scratch_bytes(8192, Mode::Full) - scratch_bytes(8192, Mode::FnSymGfx1201),
            6_443_073_536
        );
        // The 8192-row compact scratch is smaller than the 4096-row full one.
        assert!(scratch_bytes(8192, Mode::FnSymGfx1201) <= scratch_bytes(4096, Mode::Full));
        assert_eq!(
            scratch_bytes(4096, Mode::Full) - scratch_bytes(8192, Mode::FnSymGfx1201),
            2_023_577_600
        );
        // The estimator adds the same 64-row speculative logits to either plan.
        let config = fn_config();
        let spec = |rows: usize| qwen4_spec_logit_rows(rows) as u64 * (248_320 * 4 + 4);
        for (rows, mode) in [
            (4096, Mode::Full),
            (8192, Mode::Full),
            (4096, Mode::FnSymGfx1201),
            (8192, Mode::FnSymGfx1201),
        ] {
            assert_eq!(
                qwen4_forward_device_bytes(&config, rows, mode),
                Some(scratch_bytes(rows, mode) + spec(rows))
            );
        }
        assert!(
            qwen4_forward_device_bytes(&config, 8192, Mode::FnSymGfx1201)
                <= qwen4_forward_device_bytes(&config, 4096, Mode::Full)
        );
    }

    /// The allocation inventory is the owner's own plan: one entry per owning
    /// non-empty scratch slot (borrowed alias views omitted) plus the spec
    /// logits and argmax, and it sums to the byte query admission uses.
    #[test]
    fn allocation_sizes_sum_to_the_forward_device_bytes() {
        let config = fn_config();
        for mode in [Mode::Full, Mode::FnSymGfx1201] {
            for rows in [1usize, 4, 512, 768, 1536, 4096, 8192] {
                let sizes = super::qwen4_forward_allocation_sizes(&config, rows, mode);
                let bytes = qwen4_forward_device_bytes(&config, rows, mode);
                assert_eq!(
                    sizes.as_ref().map(|sizes| sizes.iter().sum::<u64>()),
                    bytes,
                    "{mode:?} {rows} rows"
                );
                let sizes = sizes.expect("representative plans admit");
                let slots = Scratch::plan(&config, rows, mode).expect("plan").0;
                let owned: Vec<u64> = slots
                    .iter()
                    .filter(|slot| slot.owner.is_none())
                    .map(|slot| slot.bytes().expect("slot bytes") as u64)
                    .filter(|&bytes| bytes > 0)
                    .collect();
                let spec_rows = qwen4_spec_logit_rows(rows) as u64;
                let mut expected = owned.clone();
                expected.push(spec_rows * config.vocab_size as u64 * 4);
                expected.push(spec_rows * 4);
                assert_eq!(sizes, expected, "{mode:?} {rows} rows");
                assert!(sizes.iter().all(|&bytes| bytes > 0));
                assert_eq!(
                    sizes[..owned.len()].iter().sum::<u64>(),
                    scratch_bytes(rows, mode)
                );
                // Aliases are views, never allocations or double charges.
                let views = slots.iter().filter(|slot| slot.owner.is_some()).count();
                assert_eq!(
                    sizes.len() + views,
                    slots.iter().filter(|slot| slot.elements > 0).count() + 2,
                    "{mode:?} {rows} rows"
                );
                if mode == Mode::Full {
                    assert_eq!(views, 0);
                } else {
                    assert!(views > 0);
                }
            }
        }
    }

    /// The MoE runtime's two pointer tables, one allocation each, summing to
    /// its byte query; a profile without experts allocates none.
    #[test]
    fn moe_runtime_allocation_sizes_are_two_pointer_tables() {
        use super::Qwen4MoeLayerRuntime as Runtime;
        let config = fn_config();
        let table = (config.num_experts * 8) as u64;
        let sizes = Runtime::allocation_sizes(&config).unwrap();
        assert_eq!(sizes, [table, table]);
        assert_eq!(Runtime::device_bytes(&config), Some(2 * table as usize));
        let mut none = config.clone();
        none.num_experts = 0;
        assert_eq!(Runtime::allocation_sizes(&none), Some(Vec::new()));
        assert_eq!(Runtime::device_bytes(&none), Some(0));
        let mut huge = config;
        huge.num_experts = usize::MAX / 8 + 1;
        assert_eq!(Runtime::allocation_sizes(&huge), None);
        assert_eq!(Runtime::device_bytes(&huge), None);
    }

    #[test]
    fn allocation_sizes_refuse_zero_rows_and_overflow_like_the_byte_query() {
        let config = fn_config();
        for mode in [Mode::Full, Mode::FnSymGfx1201] {
            assert_eq!(
                super::qwen4_forward_allocation_sizes(&config, 0, mode),
                None
            );
            assert_eq!(qwen4_forward_device_bytes(&config, 0, mode), None);
            let mut huge = config.clone();
            huge.vocab_size = usize::MAX;
            assert_eq!(super::qwen4_forward_allocation_sizes(&huge, 64, mode), None);
            assert_eq!(qwen4_forward_device_bytes(&huge, 64, mode), None);
            let mut wide = config.clone();
            wide.vocab_size = usize::MAX / 4;
            assert_eq!(
                super::qwen4_forward_allocation_sizes(&wide, 64, mode).is_some(),
                qwen4_forward_device_bytes(&wide, 64, mode).is_some()
            );
        }
    }

    #[test]
    fn compact_route_buffers_follow_the_frozen_capacities() {
        // (rows, gate/up/rot batch, expanded down, grouped gate/up, grouped down)
        for (rows, batch, expanded, gate_up, down) in [
            (
                4096usize,
                13_107_200u64,
                52_428_800u64,
                65_536_000u64,
                249_036_800u64,
            ),
            (8192, 13_107_200, 83_886_080, 114_688_000, 458_752_000),
        ] {
            let slots = plan(rows, Mode::FnSymGfx1201);
            for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
                assert_eq!(field_bytes(&slots, field), batch, "{field:?} at {rows}");
            }
            assert_eq!(field_bytes(&slots, F::MoeDownExpanded), expanded);
            assert_eq!(field_bytes(&slots, F::MoeYGateUpGrouped), gate_up);
            assert_eq!(field_bytes(&slots, F::MoeYDownGrouped), down);
        }
        // Today's plan at 8192: 200/200/200/800/437.5/875 MiB.
        let full = plan(8192, Mode::Full);
        for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
            assert_eq!(field_bytes(&full, field), 209_715_200);
        }
        assert_eq!(field_bytes(&full, F::MoeDownExpanded), 838_860_800);
        assert_eq!(field_bytes(&full, F::MoeYGateUpGrouped), 458_752_000);
        assert_eq!(field_bytes(&full, F::MoeYDownGrouped), 917_504_000);
    }

    /// Every field of every arena group lies inside the `Projection` arena at
    /// a 256-byte aligned offset, and the regions of one group are disjoint.
    fn assert_arena_groups_disjoint(config: &Qwen4Config, rows: usize, slots: &[ScratchSlot]) {
        let arena = slots[F::Projection as usize].elements;
        let groups = compact_arena_groups(config, rows, slots).unwrap();
        for group in &groups {
            let mut regions: Vec<(usize, usize, F)> = Vec::new();
            for &(field, elements, pitch) in group {
                let slot = slots[field as usize];
                if slot.elements == 0 {
                    continue;
                }
                if field != F::Projection {
                    assert_eq!(slot.owner, Some(F::Projection), "{field:?} at {rows}");
                    assert_eq!(slot.pitch, pitch, "{field:?} at {rows}");
                }
                assert_eq!(slot.offset % 64, 0, "{field:?} at {rows}");
                let extent = elements.max(pitch.map_or(0, |pitch| rows * pitch));
                assert!(slot.offset + extent <= arena, "{field:?} at {rows}");
                regions.push((slot.offset, slot.offset + extent, field));
            }
            regions.sort_by_key(|region| (region.0, region.1));
            for pair in regions.windows(2) {
                assert!(
                    pair[0].1 <= pair[1].0,
                    "{:?} overlaps {:?} at {rows}",
                    pair[0].2,
                    pair[1].2
                );
            }
        }
    }

    #[test]
    fn compact_arena_groups_are_disjoint_and_everything_else_owns_its_allocation() {
        let config = fn_config();
        let full = plan(8192, Mode::Full);
        assert_eq!(full.len(), F::COUNT);
        assert!(full
            .iter()
            .all(|slot| slot.owner.is_none() && slot.pitch.is_none() && slot.offset == 0));
        for rows in [1usize, 7, 256, 512, 4096, 16384] {
            let compact = plan(rows, Mode::FnSymGfx1201);
            assert_arena_groups_disjoint(&config, rows, &compact);
            let grouped: Vec<F> = compact_arena_groups(&config, rows, &compact)
                .unwrap()
                .into_iter()
                .flatten()
                .map(|(field, _, _)| field)
                .collect();
            for (index, slot) in compact.iter().enumerate() {
                let field = F::ALL[index];
                match field {
                    F::PleGated => assert_eq!(slot.owner, Some(F::Rotation)),
                    _ if grouped.contains(&field)
                        && field != F::Projection
                        && slot.elements != 0 =>
                    {
                        assert_eq!(slot.owner, Some(F::Projection), "{field:?} at {rows}");
                    }
                    _ => {
                        assert_eq!(slot.owner, None, "{field:?} at {rows}");
                        assert_eq!(slot.offset, 0, "{field:?} at {rows}");
                    }
                }
            }
            // The kept normed rows are dedicated; a forward wider than them
            // writes the arena view, which exists only then.
            assert_eq!(
                compact[F::PleNormed as usize].elements,
                rows.min(QWEN4_PLE_NORMED_KEEP_ROWS) * 10_240,
                "{rows}"
            );
            let wide = compact[F::PleNormedWide as usize];
            if rows > QWEN4_PLE_NORMED_KEEP_ROWS {
                assert_eq!(wide.owner, Some(F::Projection), "{rows}");
                assert_eq!(wide.pitch, Some(10_240), "{rows}");
                assert_eq!(wide.elements, rows * 10_240, "{rows}");
            } else {
                assert_eq!(wide.elements, 0, "{rows}");
            }
            for field in [
                F::Streams,
                F::HcMixed,
                F::HcGates,
                F::AttentionOutput,
                F::FfnOutput,
                F::Rotation,
                F::PleRows,
                F::PleStaged,
                F::PleNormed,
                F::TokenIds,
                F::Logits,
                F::GdnBf16,
                F::MoeTopkIndices,
                F::MoeTopkWeights,
                F::MoeExpertTokenCounts,
                F::MoeExpertOffsets,
                F::MoeExpertTileIds,
                F::MoeSortedSlotIndex,
                F::MoeInversePerm,
                F::RouterLogits,
            ] {
                assert!(!grouped.contains(&field), "{field:?}");
                assert_eq!(compact[field as usize].owner, None, "{field:?} at {rows}");
            }
            assert_eq!(
                owned_bytes(&compact),
                scratch_bytes(rows, Mode::FnSymGfx1201)
            );
            for slot in compact.iter().filter(|slot| slot.owner.is_some()) {
                let owner = compact[slot.owner.unwrap() as usize];
                assert!(owner.owner.is_none());
                if slot.dtype == rdna_compute::DType::Raw {
                    // `QsaSelected`: byte elements over an F32-covered region.
                    assert_eq!(owner.dtype, rdna_compute::DType::F32);
                    assert!(slot.offset + slot.elements.div_ceil(4) <= owner.elements);
                } else {
                    assert_eq!(owner.dtype, slot.dtype);
                    assert!(slot.offset + slot.elements <= owner.elements);
                }
            }
        }
        assert_eq!(owned_bytes(&full), scratch_bytes(8192, Mode::Full));
        let compact = plan(8192, Mode::FnSymGfx1201);
        assert_eq!(compact[F::Rotation as usize].pitch, Some(10_240));
        assert_eq!(compact[F::PleGated as usize].pitch, Some(10_240));
        assert_eq!(compact[F::Projection as usize].pitch, Some(10_240));
        assert_eq!(compact[F::PleNormed as usize].owner, None);
        // The selections are the last QSA member: a Raw byte slot.
        let selected = compact[F::QsaSelected as usize];
        assert_eq!(selected.owner, Some(F::Projection));
        assert_eq!(selected.dtype, rdna_compute::DType::Raw);
        assert_eq!(selected.pitch, None);
        assert_eq!(selected.elements, 8192 * config.qsa_selected_capacity() * 4);
    }

    /// One grouped-GDN invocation set, host-simulated: the grouped producers
    /// write `group` packed rows of `qkv` elements into the segment view of
    /// an alias-set owner (`pitch` per owner row), then the per-lane core
    /// reads row `begin + i` through the per-lane view (`begin + i` at the
    /// owner pitch) and continues with an aliased `wide`-wide write of its
    /// own row (the HC normalise / up-projection).  `Err(i)` is the first
    /// group-relative row whose packed qkv values were not where the lane
    /// read them.  Two microbatches are produced before either is consumed
    /// so a continuation write that strays into a later row is observed.
    fn simulate_grouped_gdn_rows(
        pitch: usize,
        qkv: usize,
        wide: usize,
        begin: usize,
        group: usize,
    ) -> Result<(), usize> {
        let numel = 64 * pitch;
        let mut owner = vec![0u32; numel];
        let tag = |row: usize, col: usize| 0x1000_0000u32 | ((row as u32) << 14) | col as u32;
        let microbatches = [begin, begin + group];
        // Producers: packed contiguous rows at the segment base.
        for &first in &microbatches {
            let (offset, _) = owner_pitch_extent(pitch, numel, first, group, qkv).unwrap();
            for i in 0..group {
                for c in 0..qkv {
                    owner[offset + i * qkv + c] = tag(first + i, c);
                }
            }
        }
        // Consumers: per-lane core read, then the aliased HC continuation.
        for &first in &microbatches {
            let (group_base, _) = owner_pitch_extent(pitch, numel, first, group, qkv).unwrap();
            for i in 0..group {
                let (lane_base, _) = owner_pitch_extent(pitch, numel, first + i, 1, qkv).unwrap();
                if lane_base != group_base + i * qkv
                    || (0..qkv).any(|c| owner[lane_base + c] != tag(first + i, c))
                {
                    return Err(i);
                }
                let (hc_base, _) = owner_pitch_extent(pitch, numel, first + i, 1, wide).unwrap();
                for c in 0..wide {
                    owner[hc_base + c] = 0xF000_0000u32 | ((first + i) as u32);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_gdn_qkv_rows_survive_the_compact_row_pitch() {
        let config = fn_config();
        let qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let wide = config.hidden_size * config.hc_count;
        assert_eq!(qkv, wide);
        let compact = plan(32, Mode::FnSymGfx1201);
        for field in [F::Projection, F::Projection2] {
            let pitch = compact[field as usize].pitch.expect("owner pitch");
            assert_eq!(pitch, qkv);
            for group in 1..=8usize {
                for begin in [0usize, 1, 3, 9] {
                    assert_eq!(
                        simulate_grouped_gdn_rows(pitch, qkv, wide, begin, group),
                        Ok(()),
                        "{field:?} group {group} begin {begin}"
                    );
                }
            }
        }
        // Negative control: a pitch wider than the actual qkv width (12288)
        // strands every row after the first of a group, so the odd group row
        // is the first the lane fails to find.
        for group in 1..=8usize {
            for begin in [0usize, 1, 3, 9] {
                let result = simulate_grouped_gdn_rows(12_288, qkv, wide, begin, group);
                if group == 1 {
                    assert_eq!(result, Ok(()), "group 1 begin {begin}");
                } else {
                    assert_eq!(result, Err(1), "group {group} begin {begin}");
                }
            }
        }
    }

    #[test]
    fn unequal_gdn_width_keeps_the_arena_at_the_qkv_pitch() {
        let mut config = fn_config();
        config.linear_num_key_heads = 12;
        let qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        assert_ne!(qkv, config.hidden_size * config.hc_count);
        let rows = 32usize;
        let compact = Scratch::plan(&config, rows, Mode::FnSymGfx1201).unwrap().0;
        for field in [F::Projection, F::Projection2] {
            assert_eq!(compact[field as usize].pitch, Some(qkv), "{field:?}");
        }
        assert_eq!(compact[F::Projection2 as usize].owner, Some(F::Projection));
        assert_eq!(compact[F::HcNormalized as usize].owner, Some(F::Projection));
        assert_eq!(
            owned_bytes(&compact),
            Scratch::device_bytes(&config, rows, Mode::FnSymGfx1201).unwrap()
        );
        assert_arena_groups_disjoint(&config, rows, &compact);
    }

    #[test]
    fn pitched_segments_are_bounded_by_the_arena_region_of_the_field() {
        // `segment_view` bounds a pitched field by its own `max_chunk * pitch`
        // region of the owner's allocation, which sits at the field's offset.
        let rows = 8usize;
        let config = fn_config();
        let qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let compact = plan(rows, Mode::FnSymGfx1201);
        let mut checked = 0;
        for (index, slot) in compact.iter().enumerate() {
            let Some(pitch) = slot.pitch else { continue };
            let storage = slot.owner.map_or(*slot, |owner| compact[owner as usize]);
            assert!(storage.owner.is_none(), "field {index}");
            assert!(
                slot.offset + rows * pitch <= storage.elements,
                "field {index}"
            );
            let width = match F::ALL[index] {
                F::Projection | F::Projection2 => qkv,
                _ => slot.elements / rows,
            };
            assert!(width <= pitch, "field {index}");
            let region = rows * pitch;
            let whole = owner_pitch_extent(pitch, region, 0, rows, width).unwrap();
            assert_eq!(whole, (0, rows * pitch), "field {index}");
            let last = owner_pitch_extent(pitch, region, rows - 1, 1, width).unwrap();
            assert_eq!(last, ((rows - 1) * pitch, pitch), "field {index}");
            assert!(owner_pitch_extent(pitch, region - 1, 0, rows, width).is_err());
            checked += 1;
        }
        assert!(checked >= 20);
    }

    #[test]
    fn compact_layout_holds_at_the_boundary_rows() {
        for rows in [1usize, 8, 9, 64, 511, 512, 513, 4096, 8192] {
            let full = plan(rows, Mode::Full);
            let compact = plan(rows, Mode::FnSymGfx1201);
            assert_eq!(
                owned_bytes(&full),
                scratch_bytes(rows, Mode::Full),
                "rows {rows}"
            );
            assert_eq!(
                owned_bytes(&compact),
                scratch_bytes(rows, Mode::FnSymGfx1201),
                "rows {rows}"
            );
            assert!(owned_bytes(&compact) < owned_bytes(&full), "rows {rows}");

            let n = rows as u64;
            let c = rows.min(QWEN4_COMPACT_FALLBACK_ROWS) as u64;
            for field in [F::MoeGateBatch, F::MoeUpBatch, F::MoeRotBatch] {
                assert_eq!(field_bytes(&compact, field), c * 10 * 640 * 4);
            }
            assert_eq!(
                field_bytes(&compact, F::MoeDownExpanded),
                (c * 10 * 2560 * 4).max(n * 2560 * 4).max(n * 80)
            );
            assert_eq!(
                field_bytes(&compact, F::MoeYGateUpGrouped),
                (grouped_rows(c) * 1280 * 4).max(grouped_rows(n) * 640 * 2)
            );
            let down = field_bytes(&compact, F::MoeYDownGrouped);
            assert_eq!(
                down,
                (grouped_rows(c) * 2560 * 4).max(grouped_rows(n) * 2560 * 2)
            );
            // The borrowed PLE output must fit inside the grouped-down owner.
            assert!(down >= n * 10_240 * 4, "rows {rows}");
            if rows <= QWEN4_COMPACT_FALLBACK_ROWS {
                for field in [
                    F::MoeGateBatch,
                    F::MoeUpBatch,
                    F::MoeRotBatch,
                    F::MoeDownExpanded,
                    F::MoeYGateUpGrouped,
                    F::MoeYDownGrouped,
                ] {
                    assert_eq!(
                        field_bytes(&compact, field),
                        field_bytes(&full, field),
                        "{field:?} at {rows} rows is the full F32 size"
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_plan_refuses_zero_rows_and_overflow() {
        let config = fn_config();
        for mode in [Mode::Full, Mode::FnSymGfx1201] {
            assert!(Scratch::device_bytes(&config, 0, mode).is_err());
            assert!(Scratch::device_bytes(&config, usize::MAX, mode).is_err());
            assert!(Scratch::device_bytes(&config, usize::MAX / 2, mode).is_err());
            assert_eq!(qwen4_forward_device_bytes(&config, 0, mode), None);
            assert_eq!(qwen4_forward_device_bytes(&config, usize::MAX, mode), None);
        }
    }

    #[test]
    fn compact_mode_needs_every_term() {
        // The expert-slab route: default on (unset), `1` on, `0` off; it
        // needs no symmetric proof and no IU4 request.
        let slab = CompactScratchTerms {
            knob: None,
            exact_gfx1201: true,
            grouped_gemm: true,
            sym_iu4_requested: false,
            source_symmetric: false,
            source_slab: true,
            ple_key_value_bf16: true,
            flash_next_profile: true,
        };
        assert_eq!(slab.mode(), Mode::FnSymGfx1201);
        assert_eq!(
            CompactScratchTerms {
                knob: Some(true),
                ..slab
            }
            .mode(),
            Mode::FnSymGfx1201
        );
        assert_eq!(
            CompactScratchTerms {
                sym_iu4_requested: true,
                ..slab
            }
            .mode(),
            Mode::FnSymGfx1201
        );
        assert_eq!(
            CompactScratchTerms {
                knob: Some(false),
                ..slab
            }
            .mode(),
            Mode::Full
        );
        // The symmetric IU4 route defaults on too: unset or `1`, plus the IU4
        // request plus the (assumed, GPU-checked) symmetric headers.
        let sym = CompactScratchTerms {
            knob: Some(true),
            exact_gfx1201: true,
            grouped_gemm: true,
            sym_iu4_requested: true,
            source_symmetric: true,
            source_slab: false,
            ple_key_value_bf16: true,
            flash_next_profile: true,
        };
        assert_eq!(sym.mode(), Mode::FnSymGfx1201);
        assert_eq!(
            CompactScratchTerms { knob: None, ..sym }.mode(),
            Mode::FnSymGfx1201
        );
        let sym_off = [
            CompactScratchTerms {
                knob: Some(false),
                ..sym
            },
            CompactScratchTerms {
                knob: Some(false),
                source_slab: true,
                ..sym
            },
            CompactScratchTerms {
                sym_iu4_requested: false,
                ..sym
            },
            CompactScratchTerms {
                sym_iu4_requested: false,
                ..sym
            },
            CompactScratchTerms {
                knob: None,
                source_symmetric: false,
                ..sym
            },
            CompactScratchTerms {
                source_symmetric: false,
                ..sym
            },
        ];
        for terms in sym_off {
            assert_eq!(terms.mode(), Mode::Full, "{terms:?}");
        }
        // `0` opts the symmetric route out even with every other term true.
        assert_eq!(
            CompactScratchTerms {
                knob: Some(false),
                source_slab: true,
                ..sym
            }
            .mode(),
            Mode::Full
        );
        // Device, PLE and profile terms gate both routes whatever the knob.
        for base in [
            slab,
            CompactScratchTerms {
                knob: Some(true),
                ..slab
            },
            sym,
            CompactScratchTerms { knob: None, ..sym },
        ] {
            let off = [
                CompactScratchTerms {
                    exact_gfx1201: false,
                    ..base
                },
                CompactScratchTerms {
                    grouped_gemm: false,
                    ..base
                },
                CompactScratchTerms {
                    ple_key_value_bf16: false,
                    ..base
                },
                CompactScratchTerms {
                    flash_next_profile: false,
                    ..base
                },
            ];
            for terms in off {
                assert_eq!(terms.mode(), Mode::Full, "{terms:?}");
            }
        }
    }

    #[test]
    fn compact_mode_is_exact_gfx1201_only() {
        assert!(CompactScratchTerms::is_exact_gfx1201("gfx1201"));
        for arch in [
            "gfx906", "gfx908", "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032",
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx940", "gfx941", "gfx942", "gfx12010", "gfx1201x", "gfx120", "",
        ] {
            assert!(!CompactScratchTerms::is_exact_gfx1201(arch), "{arch}");
        }
    }

    /// The host-built tag tensor: the u8 planner table is membership (cold = 1,
    /// MQ4 = 0) for W3 and E8P alike; the i32 table keeps the real codec.
    #[test]
    fn slab_codec_tag_bytes_are_membership_then_real_codecs() {
        use crate::expert_arena::{CODEC_E8P, CODEC_MQ4, CODEC_W3};
        for (cold, planner_real) in [(CODEC_W3, 1i32), (CODEC_E8P, 3i32)] {
            let codecs = [CODEC_MQ4, cold, cold, CODEC_MQ4];
            let bytes = slab_codec_tag_bytes(&codecs);
            assert_eq!(bytes.len(), 4 + 4 * 4);
            assert_eq!(&bytes[..4], &[0, 1, 1, 0]);
            let owners: Vec<i32> = bytes[4..]
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
                .collect();
            assert_eq!(owners, [0, planner_real, planner_real, 0]);
        }
        let full = slab_codec_tag_bytes(&vec![CODEC_E8P; 512]);
        assert_eq!(
            full.len(),
            hipfire_dispatch::pipeline::slab_route::CODEC_TAG_ALLOC_BYTES
        );
    }

    #[test]
    fn owner_pitch_segments_are_disjoint_and_bounded() {
        let pitch = 12_288usize;
        let numel = 8192 * pitch;
        // Many small invocations (aggregate well past 512 rows) of unequal
        // sizes, then one large segment, at nonzero row starts.
        let mut segments = Vec::new();
        let mut begin = 0usize;
        for index in 0..700usize {
            let rows = if index % 7 == 0 { 8 } else { 1 };
            segments.push((begin, rows));
            begin += rows;
        }
        segments.push((begin, 1500));
        assert!(begin + 1500 > QWEN4_COMPACT_FALLBACK_ROWS);
        for width in [12_288usize, 10_240, 6_144] {
            let mut previous_end = 0usize;
            for &(begin, rows) in &segments {
                let (offset, len) = owner_pitch_extent(pitch, numel, begin, rows, width).unwrap();
                assert_eq!(offset, begin * pitch);
                assert_eq!(len, rows * pitch);
                assert!(
                    offset >= previous_end,
                    "segment at row {begin} overlaps its predecessor"
                );
                previous_end = offset + len;
            }
            assert!(previous_end <= numel);
        }
        // Past the owner, wider than the pitch, or overflowing: refused.
        assert!(owner_pitch_extent(pitch, numel, 8191, 2, 10_240).is_err());
        assert!(owner_pitch_extent(pitch, numel, 0, 1, pitch + 1).is_err());
        assert!(owner_pitch_extent(usize::MAX, usize::MAX, 2, 1, 1).is_err());
        assert!(owner_pitch_extent(pitch, usize::MAX, usize::MAX, 1, 1).is_err());
    }

    #[test]
    fn logical_width_bases_collide_where_owner_pitch_bases_do_not() {
        // Two members of one alias set with different logical widths (a
        // 12288-wide and a 10240-wide field) serving neighbouring segments
        // A = rows 9..12 and B = row 12.
        let (a_begin, a_rows, b_begin) = (9usize, 3usize, 12usize);
        // Negative control: bases at each field's own logical width overlap.
        let a_logical = (a_begin * 12_288, (a_begin + a_rows) * 12_288);
        let b_logical_start = b_begin * 10_240;
        assert!(a_logical.0 <= b_logical_start && b_logical_start < a_logical.1);
        // Owner pitch: the same segments are disjoint.
        let pitch = 12_288usize;
        let numel = 64 * pitch;
        let (a_offset, a_len) = owner_pitch_extent(pitch, numel, a_begin, a_rows, 12_288).unwrap();
        let (b_offset, _) = owner_pitch_extent(pitch, numel, b_begin, 1, 10_240).unwrap();
        assert!(a_offset + a_len <= b_offset);
    }

    fn bytes(ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .flat_map(|&id| (id as i32).to_ne_bytes())
            .collect()
    }

    #[test]
    fn uploaded_ids_cover_sub_chunk_offsets() {
        let ids = [5u32, 6, 7, 8, 9, 10];
        let host = bytes(&ids);
        assert_eq!(uploaded_ids_range(6, &host, 0, &ids), Some((0, 24)));
        assert_eq!(uploaded_ids_range(6, &host, 2, &ids[2..5]), Some((8, 12)));
        assert_eq!(uploaded_ids_range(6, &host, 5, &ids[5..]), Some((20, 4)));
    }

    #[test]
    fn uploaded_ids_refuse_unclaimed_or_out_of_bounds_rows() {
        let ids = [5u32, 6, 7, 8];
        let host = bytes(&ids);
        // Nothing claimed (HIP single-row body, device argmax, graph capture).
        assert_eq!(uploaded_ids_range(0, &host, 0, &ids), None);
        // The range may not pass the uploaded rows.
        assert_eq!(uploaded_ids_range(4, &host, 0, &[5, 6, 7, 8, 9]), None);
        assert_eq!(uploaded_ids_range(4, &host, 3, &[8, 9]), None);
        assert_eq!(uploaded_ids_range(3, &host, 2, &[7, 8]), None);
        // The host mirror must hold the claimed rows.
        assert_eq!(uploaded_ids_range(4, &host[..12], 0, &ids), None);
        assert_eq!(
            uploaded_ids_range(usize::MAX, &host, usize::MAX, &[1]),
            None
        );
        assert_eq!(uploaded_ids_range(4, &host, 0, &[]), None);
    }

    #[test]
    fn uploaded_ids_refuse_different_tokens() {
        let ids = [5u32, 6, 7, 8];
        let host = bytes(&ids);
        assert_eq!(uploaded_ids_range(4, &host, 0, &[5, 6, 7, 9]), None);
        assert_eq!(uploaded_ids_range(4, &host, 1, &[5, 6]), None);
    }

    /// The pre-profile scratch plan, spelled out from the pinned scalars:
    /// the Flash-Next byte accounting must not move.
    fn flash_next_plan_bytes(config: &crate::config::Qwen4Config, r: usize) -> u64 {
        let (h, hc) = (config.hidden_size, config.hc_count);
        let w = h * hc;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let gdn_value = config.linear_num_value_heads * config.linear_value_head_dim;
        let gdn_qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim + gdn_value;
        let index = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        let max_projection = w.max(2 * q).max(gdn_qkv).max(h);
        let max_rotation = w.max(h).max(config.moe_intermediate_size);
        let (k, e, mi, shared) = (
            config.num_experts_per_tok,
            config.num_experts,
            config.moe_intermediate_size,
            config.shared_expert_intermediate_size,
        );
        let slots = r * k;
        let grouped = crate::program::grouped_m_total_bound(slots, e).unwrap();
        let selected = r * config.qsa_selected_capacity() * 4;
        let f32_elements = r * h * 2
            + r * w * 2
            + r * config.hc_lowrank
            + r * w
            + r * h
            + r * hc
            + r * max_rotation
            + r * max_projection * 2
            + r * config.linear_num_value_heads * 4
            + r * gdn_value * 3
            + r * index
            + r * 2 * q
            + r * kv * 2
            + r * q
            + r * h
            + r * h
            + r * w
            + r * h
            + r * w * 4
            + r * e
            + r * h
            + r * 2 * mi
            + r * shared * 3
            + r * h * 2
            + slots * mi * 3
            + r * k
            + slots * h
            + r * shared
            + grouped * 2 * mi
            + grouped * h
            + config.vocab_size;
        let raw_bytes = r * 4
            + selected
            + slots * 4
            + e * 4
            + (e + 1) * 4
            + grouped * 4
            + grouped / 16 * 4
            + slots * 4;
        let bf16_bytes = 2 * gdn_value + 2 * r * 16 * 160;
        (f32_elements * 4 + raw_bytes + bf16_bytes) as u64
    }

    #[test]
    fn flash_next_scratch_bytes_are_unchanged() {
        let config = crate::config::compact_test_config();
        for rows in [1, 16, 1536] {
            assert_eq!(
                super::Qwen4GpuForwardScratch::device_bytes(
                    &config,
                    rows,
                    super::Qwen4ForwardScratchMode::Full
                )
                .unwrap(),
                flash_next_plan_bytes(&config, rows),
                "rows {rows}"
            );
        }
    }

    fn tiny_dense() -> crate::config::Qwen4Config {
        crate::config::synthetic_test_config(
            crate::config::Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            true,
        )
    }

    #[test]
    fn dense_scratch_is_profile_derived_and_never_empty() {
        let config = tiny_dense();
        let rows = 4;
        let (plan, ple_bytes) = super::Qwen4GpuForwardScratch::plan(
            &config,
            rows,
            super::Qwen4ForwardScratchMode::Full,
        )
        .unwrap();
        // `Full` plans no compact-only `PleNormedWide`.
        assert!(plan.iter().enumerate().all(|(index, slot)| {
            slot.elements >= 1
                || super::ScratchField::ALL[index] == super::ScratchField::PleNormedWide
        }));
        let ple = config.ple.as_ref().unwrap();
        let embed = ple.head_count * ple.row_width;
        assert_eq!(ple_bytes, rows * embed * 2);
        use super::ScratchField as F;
        let slot = |field: F| (plan[field as usize].elements, plan[field as usize].dtype);
        let gdn_value = config.linear_num_value_heads * config.linear_value_head_dim;
        assert_eq!(
            slot(F::GdnBf16),
            (
                gdn_value.max(config.hidden_size).max(config.hc_lowrank),
                rdna_compute::DType::BF16
            )
        );
        assert_eq!(
            slot(F::PleStaged),
            (rows * embed, rdna_compute::DType::BF16)
        );
        assert_eq!(
            slot(F::PleKey).0,
            rows * config.hidden_size * config.hc_count
        );
        assert_eq!(slot(F::MoeGateBatch).0, rows * 8);
        assert_eq!(slot(F::MoeUpBatch).0, rows * 8);
    }

    #[test]
    fn forward_refuses_a_rope_base_the_qsa_kernels_do_not_implement() {
        let mut config = tiny_dense();
        let error = super::Qwen4GpuForward::admit_profile(&config).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains(&config.rope_theta.to_string())
                && message.contains(&super::QWEN4_QSA_KERNEL_ROPE_THETA.to_string()),
            "{message}"
        );
        config.rope_theta = super::QWEN4_QSA_KERNEL_ROPE_THETA;
        super::Qwen4GpuForward::admit_profile(&config).expect("kernel base admitted");
        // PLE is optional: a profile without it is admitted, and its scratch
        // plan holds no PLE staging bytes (every buffer still has an element).
        config.ple = None;
        super::Qwen4GpuForward::admit_profile(&config).expect("no PLE is admitted");
        let (plan, ple_bytes) =
            super::Qwen4GpuForwardScratch::plan(&config, 4, super::Qwen4ForwardScratchMode::Full)
                .unwrap();
        assert_eq!(ple_bytes, 0);
        // `Full` plans no compact-only `PleNormedWide`.
        assert!(plan.iter().enumerate().all(|(index, slot)| {
            slot.elements >= 1
                || super::ScratchField::ALL[index] == super::ScratchField::PleNormedWide
        }));
    }

    #[test]
    fn dense_ffn_program_dims_and_layout() {
        let config = tiny_dense();
        let dims = super::program_dims(&config);
        assert!(matches!(
            dims.ffn,
            crate::program::Qwen4FfnDims::Dense { intermediate: 8 }
        ));
        let layout = crate::program::Qwen4ScratchLayout::for_rows(dims, 4).unwrap();
        assert_eq!(layout.dense_rows, 32);
        assert_eq!(
            (
                layout.slots,
                layout.grouped_rows,
                layout.shared_rows,
                layout.routed_rows
            ),
            (0, 0, 0, 0)
        );
        let flash = super::program_dims(&crate::config::compact_test_config());
        let layout = crate::program::Qwen4ScratchLayout::for_rows(flash, 4).unwrap();
        assert_eq!(layout.dense_rows, 0);
        assert_eq!(layout.routed_rows, 4 * 10 * 640);
    }
}

/// The session cache's next pending snapshot boundary and the one after it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SessionBoundaries {
    pub next: Option<usize>,
    pub after: Option<usize>,
}

/// A shared deterministic singleton/MTP schedule. A planned pair covers
/// `min(remaining, 2 * chunk)` rows. One interior session boundary rides the
/// in-bank Core sub-launch; a second interior boundary ends the group at the
/// first one, as an ordinary sequential bank.
pub(crate) fn qwen4_prefill_segment_rows(
    position: usize,
    prompt_end: usize,
    chunk: usize,
    boundaries: SessionBoundaries,
    planned: bool,
) -> usize {
    let remaining = prompt_end.saturating_sub(position);
    let pair = planned && chunk == 8192 && remaining > chunk;
    let rows = if pair {
        remaining.min(2 * chunk)
    } else {
        remaining.min(chunk)
    };
    let Some(boundary) = boundaries.next else {
        return rows;
    };
    if boundary <= position {
        return remaining.min(chunk);
    }
    let distance = boundary - position;
    if pair {
        if distance < rows && boundaries.after.is_some_and(|next| next < position + rows) {
            return distance.min(chunk);
        }
        return rows;
    }
    rows.min(distance)
}

/// Where in a pair group the session snapshot row falls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CutSite {
    /// No interior snapshot row (the group end is published by the driver).
    None,
    /// Exactly the A/B boundary: bank A runs whole, state staged after it.
    Split,
    /// Inside `bank` at its local `row`: that bank's Core runs as two launches.
    Bank { bank: usize, row: usize },
}

/// Bank and sub-launch geometry of one pair group: two banks and the group-local
/// row of an interior session snapshot. Both the executor and the numerical twin
/// derive their chunking from this, so they cut at identical rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PairShape {
    pub rows: [usize; 2],
    pub cut: Option<usize>,
}

impl PairShape {
    /// `cut` is the session boundary `boundary` made group-local at `position`;
    /// only a boundary strictly inside the group is a cut.
    pub(crate) fn new(rows: [usize; 2], position: usize, boundary: Option<usize>) -> Self {
        let total = rows[0] + rows[1];
        let cut = boundary
            .filter(|&b| b > position && b < position + total)
            .map(|b| b - position);
        Self { rows, cut }
    }

    pub(crate) fn total(&self) -> usize {
        self.rows[0] + self.rows[1]
    }

    pub(crate) fn site(&self) -> CutSite {
        match self.cut {
            None => CutSite::None,
            Some(cut) if cut == self.rows[0] => CutSite::Split,
            Some(cut) if cut < self.rows[0] => CutSite::Bank { bank: 0, row: cut },
            Some(cut) => CutSite::Bank {
                bank: 1,
                row: cut - self.rows[0],
            },
        }
    }

    /// Group-local row ranges of the sequential chunks the numerical twin runs:
    /// the banks, further split at the cut (a sub-launch is a chunk boundary).
    pub(crate) fn pieces(&self) -> Vec<std::ops::Range<usize>> {
        let mut points = vec![0, self.rows[0], self.total()];
        points.extend(self.cut);
        points.sort_unstable();
        points.dedup();
        points.windows(2).map(|w| w[0]..w[1]).collect()
    }
}

#[cfg(test)]
mod ep_rank_generic_tests {
    use super::*;

    /// Default on, exact `0` is the kill switch, anything else leaves it on.
    #[test]
    fn ep_levers_are_default_on_with_an_exact_zero_kill_switch() {
        assert!(ep_lever_on(None));
        assert!(ep_lever_on(Some("1")));
        assert!(ep_lever_on(Some("")));
        assert!(ep_lever_on(Some("true")));
        assert!(!ep_lever_on(Some("0")));
    }

    #[test]
    fn owner_routed_prefill_admits_exactly_ep2_through_ep8_on_gfx1100_and_gfx1201() {
        for ranks in 2usize..=8 {
            assert!(tile_ranks_supported(ranks), "{ranks}");
        }
        for ranks in [0usize, 1, 9, 16] {
            assert!(!tile_ranks_supported(ranks), "{ranks}");
        }
        for arch in ["gfx1100", "gfx1201"] {
            assert!(rdna_compute::moe::ep_tile_pm_arch(arch), "{arch}");
        }
        for arch in ["gfx1200", "gfx1151", "gfx1101", "gfx942", ""] {
            assert!(!rdna_compute::moe::ep_tile_pm_arch(arch), "{arch:?}");
        }
    }

    #[test]
    fn stage_donor_bytes_split_by_every_rank_and_total_the_full_layer() {
        let full = 512 * (QWEN4_EXPERT_STAGE_GATE_UP_BYTES + QWEN4_EXPERT_STAGE_DOWN_BYTES);
        for ranks in [1usize, 2, 3, 4, 5, 6, 8] {
            let mut total = 0;
            for rank in 0..ranks {
                let count = Qwen4EpPlacement { rank, ranks }.local_count(512);
                let [gate_up, down] = stage_donor_bytes(512, rank, ranks);
                assert_eq!(gate_up, count * QWEN4_EXPERT_STAGE_GATE_UP_BYTES);
                assert_eq!(down, count * QWEN4_EXPERT_STAGE_DOWN_BYTES);
                total += gate_up + down;
            }
            assert_eq!(total, full, "ranks={ranks}");
        }
    }

    #[test]
    fn donor_redirect_covers_each_expert_once_and_keeps_peer_pointers() {
        for ranks in [2usize, 3, 4, 5, 6, 8] {
            donor_redirect_case(ranks);
        }
    }

    fn donor_redirect_case(ranks: usize) {
        let stride = QWEN4_EXPERT_STAGE_GATE_UP_BYTES;
        let base = 0x10_0000usize;
        let entries: Vec<usize> = (0..512).map(|e| 0x9000_0000 + e).collect();
        let mut claimed = vec![0usize; 512];
        for rank in 0..ranks {
            let placement = Qwen4EpPlacement { rank, ranks };
            let bytes = redirect_owned_entries(placement, base, stride, &entries);
            for (expert, word) in bytes.chunks_exact(std::mem::size_of::<usize>()).enumerate() {
                let pointer = usize::from_ne_bytes(word.try_into().unwrap());
                if expert % ranks == rank {
                    let slot = expert / ranks;
                    // The donor holds `local_count` slots; the owned slot is inside it.
                    assert!(
                        slot < placement.local_count(512),
                        "rank {rank} expert {expert}"
                    );
                    assert_eq!(pointer, base + slot * stride, "rank {rank} expert {expert}");
                    claimed[expert] += 1;
                } else {
                    assert_eq!(pointer, entries[expert], "rank {rank} expert {expert}");
                }
            }
        }
        assert!(claimed.iter().all(|&count| count == 1));
    }
}

#[cfg(test)]
mod mesh_range_tests {
    use super::*;
    use crate::config::compact_test_config;
    use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
    use hipfire_runtime::mesh_placement::BandPlacement;

    #[test]
    fn pp1_moe_runtime_slots_visit_every_layer_and_find_each_slab() {
        use crate::weights::{Qwen4Manifest, Qwen4TierRecipe};
        use hipfire_runtime::weight_store::{
            WeightLoadTransaction, WeightProjection, WeightProjectionKind, WeightStore,
        };

        let config = compact_test_config();
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 1)]).unwrap();
        let placement = BandPlacement::from_counts(
            &mesh,
            config.num_hidden_layers,
            &[config.num_hidden_layers],
        )
        .unwrap();
        let scope = Qwen4BandScope::from_placement(&config, &placement, 0).unwrap();
        assert_eq!(scope, Qwen4BandScope::single(&config));
        let manifest = Qwen4Manifest::build_with_recipe(
            &config,
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: DType::Q8_0,
                mtp_source: false,
            },
        )
        .unwrap()
        .with_band_scope(&config, &scope)
        .unwrap();
        let placements = manifest.placements_for_scope(&config, &scope, 0).unwrap();
        // Use the real CPU symbolic-ownership assembly seam, as in the
        // weights tests: no GPU allocation or fabricated runtime/tensor.
        let mut store = WeightStore::default();
        for entry in manifest
            .weights
            .iter()
            .filter(|entry| !entry.residency.is_external())
        {
            store
                .stage_alias(
                    &entry.name,
                    entry.layer,
                    0,
                    "test-source",
                    WeightProjection {
                        kind: WeightProjectionKind::Static,
                        axis: None,
                        rank: 0,
                        world_size: 1,
                        logical_shape: entry.logical_shape.clone(),
                        dtype: entry.dtype,
                    },
                )
                .unwrap();
        }
        let mut tx = WeightLoadTransaction::new(store);
        let weights = Qwen4Weights::assemble(
            &mut tx,
            &config,
            &manifest.identity.recipe,
            &placements,
            &scope,
        )
        .unwrap();
        assert!(tx.is_empty());
        assert!(weights.ep_placement().is_none());
        let moe_layers = weights
            .layer_refs
            .iter()
            .filter(|layer| layer.ffn.moe().is_some())
            .count();
        assert_eq!(moe_layers, config.num_hidden_layers);
        let mut slots = vec![None; config.num_hidden_layers];
        let mut visited = Vec::new();
        populate_moe_runtimes(&mut slots, &weights.layer_refs, |layer, _| {
            visited.push(layer);
            Ok::<_, std::convert::Infallible>(layer)
        })
        .unwrap();
        let expected: Vec<_> = (0..config.num_hidden_layers).collect();
        assert_eq!(
            visited, expected,
            "the actual PP1 builder selects every MoE layer"
        );
        assert!(slots.iter().all(Option::is_some));
        assert_eq!(slots.iter().flatten().count(), moe_layers);
        assert_eq!(
            slots.iter().flatten().copied().collect::<Vec<_>>(),
            expected
        );
        // Put the only slab at each actual layer in turn. In particular a
        // slab at the last layer must still trigger the reserve's any walk.
        for slab_layer in expected {
            assert!(
                slots.iter().flatten().any(|&layer| layer == slab_layer),
                "reserve skipped sole slab layer {slab_layer}"
            );
        }
        let absent = [None::<usize>; 3];
        let empty: [Option<usize>; 0] = [];
        for slots in [&absent[..], &empty[..]] {
            assert_eq!(slots.iter().flatten().count(), 0);
            assert!(!slots.iter().flatten().any(|_| true));
        }
    }

    #[test]
    fn trunk_op_ranges_tile_the_graph_and_the_head_trails_the_output_band() {
        let config = compact_test_config();
        let full =
            crate::program::qwen4_step_graph_with_output(&config, MeshRoute::Decode, true, true)
                .unwrap();
        let total = full.ops().len() as u32;
        for counts in [vec![24, 24], vec![9, 8, 8, 8, 8, 7]] {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, counts.len())]).unwrap();
            let placement = BandPlacement::from_counts(&mesh, config.num_hidden_layers, &counts)
                .unwrap()
                .with_first_band_min(2)
                .unwrap();
            let mut next = 0u32;
            for stage in 0..counts.len() {
                let scope = Qwen4BandScope::from_placement(&config, &placement, stage).unwrap();
                let trunk = mesh_trunk_op_range(&config, &scope).unwrap();
                assert_eq!(trunk.start, next, "bands tile the trunk ops without a gap");
                next = trunk.end;
                let ids = crate::program::qwen4_rank_op_ids(&full, &scope);
                let head_ops = if scope.output { 3 } else { 0 };
                assert_eq!(ids.len() as u32, trunk.end - trunk.start + head_ops);
                assert_eq!(ids.first().map(|id| id.0), Some(trunk.start));
            }
            assert_eq!(next + 3, total, "the head ops trail the last trunk op");
        }
    }
}

#[cfg(test)]
mod band_scratch_tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::expert_residency::forward_census_scope;
    use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
    use hipfire_runtime::mesh_placement::BandPlacement;

    /// The scratch a band plans plus the output band's speculative rows is
    /// exactly the S5 forward census of that band (both modes; a geometry the
    /// census refuses the plan refuses too), and the whole-model scope is the
    /// historical plan.
    #[test]
    fn band_plan_bytes_equal_the_forward_census() {
        let config = compact_test_config();
        let single = Qwen4BandScope::single(&config);
        for mode in [
            Qwen4ForwardScratchMode::Full,
            Qwen4ForwardScratchMode::FnSymGfx1201,
        ] {
            for rows in [64usize, 256] {
                assert_eq!(
                    Qwen4GpuForwardScratch::device_bytes_scoped(&config, rows, mode, &single).ok(),
                    Qwen4GpuForwardScratch::device_bytes(&config, rows, mode).ok()
                );
                for counts in [vec![24, 24], vec![9, 8, 8, 8, 8, 7]] {
                    let mesh = DeviceMesh::rect(&[(DimKind::Pp, counts.len())]).unwrap();
                    let placement =
                        BandPlacement::from_counts(&mesh, config.num_hidden_layers, &counts)
                            .unwrap()
                            .with_first_band_min(2)
                            .unwrap();
                    for stage in 0..counts.len() {
                        let scope =
                            Qwen4BandScope::from_placement(&config, &placement, stage).unwrap();
                        let planned = Qwen4GpuForwardScratch::device_bytes_scoped(
                            &config, rows, mode, &scope,
                        );
                        let census = forward_census_scope(&config, rows, mode, &scope)
                            .and_then(|census| census.total());
                        match (planned, census) {
                            (Ok(planned), Ok(census)) => {
                                let spec = if scope.output {
                                    (qwen4_spec_logit_rows(rows) * (config.vocab_size * 4 + 4))
                                        as u64
                                } else {
                                    0
                                };
                                assert_eq!(
                                    planned + spec,
                                    census,
                                    "{mode:?} {rows} rows stage {stage}"
                                );
                            }
                            (Err(_), Err(_)) => {}
                            (planned, census) => {
                                panic!("plan {planned:?} and census {census:?} disagree")
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A layer's table and cache are bound to the runtime `layer_desc` bound, and
/// the seal then compares that source identity with the routed experts the
/// call carries. These cases run the forward's own `select_layer_moe` and
/// `bind_routed_view` over CPU fixtures standing in for the static, EP-full and
/// balanced-live runtimes, with the view closure applying the unchanged
/// `d2_executor_admitted` (production passes `Qwen4MoeLayerRuntime::routed_view`).
/// The fixtures have distinct source identities, so equal identity here means
/// "the view of the runtime the layer was bound to"; the real seal may also
/// accept a distinct source whose per-expert pointers are identical.
#[cfg(test)]
mod seal_view_identity_tests {
    use super::*;

    /// An executor-less view: the runtime's identity, a distinct address.
    struct FakeSet {
        id: u64,
    }

    impl RoutedExpertWeights for FakeSet {
        fn len(&self) -> usize {
            4
        }

        fn get(&self, _expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
            None
        }

        fn immutable_identity(&self) -> Option<u64> {
            Some(self.id)
        }
    }

    /// A layer runtime: `id` is its immutable source identity.
    struct FakeRuntime {
        id: u64,
        split: bool,
        withheld: FakeSet,
    }

    impl FakeRuntime {
        fn new(id: u64, split: bool) -> Self {
            Self {
                id,
                split,
                withheld: FakeSet { id },
            }
        }

        /// The fixture's `routed_view`: the same decision, the fixture's
        /// executor-less view.
        fn view(
            &self,
            d2_phase: bool,
            rows: usize,
            route_rows: usize,
            lab_armed: bool,
            verify_rows: usize,
        ) -> &dyn RoutedExpertWeights {
            let capacity = if self.split { 64 } else { 0 };
            if d2_executor_admitted(
                self.split,
                d2_phase,
                rows,
                route_rows,
                lab_armed,
                verify_rows,
                capacity,
            ) {
                return self;
            }
            &self.withheld
        }
    }

    impl RoutedExpertWeights for FakeRuntime {
        fn len(&self) -> usize {
            4
        }

        fn get(&self, _expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
            None
        }

        fn immutable_identity(&self) -> Option<u64> {
            Some(self.id)
        }
    }

    const STATIC_ID: u64 = 0x51;
    const FULL_ID: u64 = 0xF1;
    const LIVE_ID: u64 = 0xB1;

    fn address(view: &dyn RoutedExpertWeights) -> *const () {
        view as *const dyn RoutedExpertWeights as *const ()
    }

    struct Runtimes {
        static_moe: FakeRuntime,
        full: FakeRuntime,
        live: FakeRuntime,
    }

    impl Runtimes {
        fn new(split: bool) -> Self {
            Self {
                static_moe: FakeRuntime::new(STATIC_ID, split),
                full: FakeRuntime::new(FULL_ID, split),
                live: FakeRuntime::new(LIVE_ID, split),
            }
        }

        /// What the forward does for one layer: select, then bind the seal
        /// view. Returns the selected runtime and the bound routed view.
        #[allow(clippy::too_many_arguments)]
        fn bind(
            &self,
            ep: bool,
            balanced: bool,
            full_chunk: bool,
            tile_route: bool,
            d2_phase: bool,
            rows: usize,
            route_rows: usize,
        ) -> (&FakeRuntime, &dyn RoutedExpertWeights) {
            self.bind_scoped(
                ep, balanced, full_chunk, tile_route, d2_phase, rows, route_rows, false, 8,
            )
        }

        /// Same with an explicit lab scope (presence + frozen verify rows).
        fn bind_scoped(
            &self,
            ep: bool,
            balanced: bool,
            full_chunk: bool,
            tile_route: bool,
            d2_phase: bool,
            rows: usize,
            route_rows: usize,
            lab_armed: bool,
            verify_rows: usize,
        ) -> (&FakeRuntime, &dyn RoutedExpertWeights) {
            let selected = select_layer_moe(
                balanced.then_some(&self.live),
                ep.then_some(Some(&self.full)),
                full_chunk,
                tile_route,
                Some(&self.static_moe),
            );
            let described: &dyn RoutedExpertWeights = selected.expect("a runtime is selected");
            let view = bind_routed_view(selected, described, |runtime| {
                runtime.view(d2_phase, rows, route_rows, lab_armed, verify_rows)
            });
            (selected.unwrap(), view)
        }
    }

    /// EP few-row gathered forward (decode, verify): the balanced live runtime
    /// is selected and its own view is bound, D2 armed or not.
    #[test]
    fn ep_balanced_decode_binds_the_live_runtime_view() {
        for split in [false, true] {
            let rt = Runtimes::new(split);
            for rows in 1..=8usize {
                let (selected, view) = rt.bind(true, true, false, false, true, rows, rows);
                assert_eq!(selected.id, LIVE_ID);
                assert_eq!(view.immutable_identity(), Some(selected.id));
                assert_ne!(view.immutable_identity(), Some(STATIC_ID));
                assert_ne!(view.immutable_identity(), Some(FULL_ID));
            }
        }
    }

    /// EP full chunk (a first AR warmup between the few-row cap and a
    /// prefill tile, or a prefill chunk): the EP full runtime is selected, and
    /// the D2 withholding (wide window, prefill phase) is a view of it too.
    #[test]
    fn ep_full_chunk_binds_the_full_runtime_view_through_d2_withholding() {
        for split in [false, true] {
            let rt = Runtimes::new(split);
            for (d2_phase, rows, route_rows) in [
                (true, 16, 16),
                (true, 511, 511),
                (false, 16, 16),
                (false, 2048, 2048),
                (true, 3, 8192),
                (false, 2, 2),
            ] {
                let (selected, view) =
                    rt.bind(true, false, true, false, d2_phase, rows, route_rows);
                assert_eq!(selected.id, FULL_ID);
                assert_eq!(view.immutable_identity(), Some(selected.id));
                assert_ne!(view.immutable_identity(), Some(STATIC_ID));
                // Withheld exactly when the unchanged D2 gate refuses it.
                assert_eq!(
                    address(view) != address(selected),
                    !d2_executor_admitted(split, d2_phase, rows, route_rows, false, 8, 4)
                );
            }
        }
    }

    /// An EP forward of a few non-gathered rows, or a tile-route forward, binds
    /// the static runtime and its view, never the EP tables.
    #[test]
    fn ep_non_gathered_few_rows_and_tile_route_bind_the_static_view() {
        for split in [false, true] {
            let rt = Runtimes::new(split);
            let (selected, view) = rt.bind(true, false, false, false, true, 4, 4);
            assert_eq!(selected.id, STATIC_ID);
            assert_eq!(view.immutable_identity(), Some(selected.id));
            let (selected, view) = rt.bind(true, false, true, true, false, 64, 64);
            assert_eq!(selected.id, STATIC_ID);
            assert_eq!(view.immutable_identity(), Some(selected.id));
        }
    }

    /// The pre-fix override viewed the static runtime whatever the layer
    /// selected; its source identity differs from an EP-selected runtime's.
    #[test]
    fn a_static_runtime_view_carries_a_different_source_than_an_ep_selected_runtime() {
        let rt = Runtimes::new(true);
        for (balanced, full_chunk, id) in [(true, false, LIVE_ID), (false, true, FULL_ID)] {
            let (selected, _) = rt.bind(true, balanced, full_chunk, false, true, 4, 4);
            assert_eq!(selected.id, id);
            let stale = rt.static_moe.view(true, 4, 4, false, 8);
            assert_ne!(stale.immutable_identity(), Some(selected.id));
        }
    }

    /// Single-card and pipeline-parallel forwards have no EP: the static
    /// runtime is bound and the D2 gate behaves as before: armed admits the
    /// runtime itself only for a whole narrow non-prefill window, off always.
    #[test]
    fn single_card_and_pp_keep_the_static_runtime_and_the_d2_gate() {
        for split in [false, true] {
            let rt = Runtimes::new(split);
            for (d2_phase, rows, route_rows) in [
                (true, 1, 1),
                (true, 4, 4),
                (true, 5, 5),
                (true, 16, 16),
                (false, 1, 1),
                (false, 4, 4),
                (true, 2, 8192),
                (false, 2048, 2048),
            ] {
                let (selected, view) =
                    rt.bind(false, false, rows > 1, false, d2_phase, rows, route_rows);
                assert_eq!(selected.id, STATIC_ID);
                assert_eq!(view.immutable_identity(), Some(selected.id));
                let admitted = d2_executor_admitted(split, d2_phase, rows, route_rows, false, 8, 4);
                assert_eq!(address(view) == address(selected), admitted);
                if !split {
                    // Nothing armed: never withheld.
                    assert_eq!(address(view), address(selected));
                }
            }
        }
    }

    /// Lab scope (wiring only, not numeric proof): a whole logical 1..64-row
    /// window outside a prefill tile binds the runtime itself when the lab is
    /// armed with capacity; tails, prefill-phase calls and unarmed scopes bind
    /// the withheld view.
    #[test]
    fn lab_scope_binds_whole_1_to_64_windows_wiring_only() {
        for split in [false, true] {
            let rt = Runtimes::new(split);
            for rows in 1..=64usize {
                let (selected, view) =
                    rt.bind_scoped(false, false, rows > 1, false, true, rows, rows, true, 64);
                assert_eq!(view.immutable_identity(), Some(selected.id));
                let admitted = d2_executor_admitted(split, true, rows, rows, true, 64, 64);
                assert_eq!(address(view) == address(selected), admitted);
                // Tails and prefill-phase calls stay withheld even when armed.
                let (selected, tail) =
                    rt.bind_scoped(false, false, true, false, true, rows, 8192, true, 64);
                assert_eq!(
                    address(tail) == address(selected),
                    d2_executor_admitted(split, true, rows, 8192, true, 64, 64)
                );
                let (selected, prefill) =
                    rt.bind_scoped(false, false, true, false, false, rows, rows, true, 64);
                assert_eq!(
                    address(prefill) == address(selected),
                    d2_executor_admitted(split, false, rows, rows, true, 64, 64)
                );
            }
        }
    }

    /// Without a selected runtime the description's own binding stands.
    #[test]
    fn no_selected_runtime_keeps_the_described_binding() {
        let described = FakeSet { id: 0x77 };
        let view = bind_routed_view::<FakeRuntime>(None, &described, |runtime| {
            runtime.view(true, 1, 1, false, 8)
        });
        assert_eq!(address(view), address(&described));
    }

    /// A lane MoE invocation binds exactly what the singleton's gathered
    /// forward (1 row decode, 2..=8 row few-row) selects and views, balanced
    /// or not, D2 armed or not; never the EP full runtime.
    #[test]
    fn ep_lane_moe_selects_and_views_the_singleton_gathered_runtime() {
        for split in [false, true] {
            let rt = Runtimes::new(split);
            for balanced in [false, true] {
                for rows in 1..=QWEN4_EP_GATHER_MAX_ROWS {
                    let (singleton, singleton_view) =
                        rt.bind(true, balanced, false, false, true, rows, rows);
                    let lane = select_ep_lane_moe(
                        balanced.then_some(&rt.live),
                        Some(Some(&rt.full)),
                        Some(&rt.static_moe),
                    )
                    .expect("a runtime is selected");
                    assert!(std::ptr::eq(lane, singleton));
                    assert_ne!(lane.id, FULL_ID);
                    assert_eq!(lane.id, if balanced { LIVE_ID } else { STATIC_ID });
                    let described: &dyn RoutedExpertWeights = lane;
                    let view = bind_routed_view(Some(lane), described, |runtime| {
                        runtime.view(true, rows, rows, false, 8)
                    });
                    assert_eq!(address(view), address(singleton_view));
                    assert_eq!(view.immutable_identity(), Some(lane.id));
                }
            }
        }
    }

    /// Outside EP the lane selection is the static runtime (unchanged).
    #[test]
    fn lane_moe_outside_ep_is_the_static_runtime() {
        let rt = Runtimes::new(false);
        let lane = select_ep_lane_moe(None, None, Some(&rt.static_moe)).unwrap();
        assert!(std::ptr::eq(lane, &rt.static_moe));
        assert!(select_ep_lane_moe::<FakeRuntime>(None, None, None).is_none());
    }

    /// Rows 1..=8 with equal routing rows pass; over the cap, zero or a
    /// re-routed count refuse.
    #[test]
    fn ep_lane_rows_refuse_over_cap_and_rerouted() {
        for rows in 1..=QWEN4_EP_GATHER_MAX_ROWS {
            assert!(ep_lane_moe_rows(rows, rows).is_ok());
            assert_eq!(ep_lane_shared_down(rows), rows >= 2);
        }
        assert!(ep_lane_moe_rows(0, 0).is_err());
        assert!(
            ep_lane_moe_rows(QWEN4_EP_GATHER_MAX_ROWS + 1, QWEN4_EP_GATHER_MAX_ROWS + 1).is_err()
        );
        assert!(ep_lane_moe_rows(1, 2).is_err());
        assert!(ep_lane_moe_rows(4, 1).is_err());
    }

    /// The buffer sequence of the emitted MoE invocations: lane-major (every
    /// layer of lane 0, then lane 1, ...; the whole program one per-lane run)
    /// and layer-major (every segment of layer 0, then layer 1, ...; grouped
    /// or mixed shares), over 1..=8 lanes and 48 layers, only ever 0/1,
    /// alternating (never the same index twice in a row), also across two
    /// consecutive steps.
    #[test]
    fn ep_lane_buffers_alternate_over_every_emission_order() {
        const LAYERS: usize = 48;
        for lanes in 1..=8usize {
            for lane_major in [true, false] {
                let invocations = LAYERS * lanes;
                assert!(ep_lane_check_invocations(invocations).is_ok());
                let mut last: Option<usize> = None;
                for _step in 0..2 {
                    let mut ordinal = 0usize;
                    for layer_or_lane in 0..invocations {
                        // Emission order only decides which (lane, layer) owns
                        // an ordinal; the buffer depends on the ordinal alone.
                        let _owner = if lane_major {
                            (layer_or_lane / LAYERS, layer_or_lane % LAYERS)
                        } else {
                            (layer_or_lane % lanes, layer_or_lane / lanes)
                        };
                        let buffer = ep_lane_buffer(ordinal);
                        ordinal += 1;
                        assert!(buffer < 2);
                        assert_ne!(last, Some(buffer), "lanes={lanes} lane_major={lane_major}");
                        last = Some(buffer);
                    }
                    assert_eq!(ordinal, invocations);
                    assert_eq!(last, Some(1));
                }
            }
        }
        // Grouped stages emit fewer, wider invocations per layer.
        for groups in 1..=8usize {
            assert!(ep_lane_check_invocations(LAYERS * groups).is_ok());
        }
        assert!(ep_lane_check_invocations(LAYERS * 3 + 1).is_err());
    }

    /// The lane buffer is not the singleton's layer parity once a layer
    /// repeats (lane-major second lane starts at layer 0 again).
    #[test]
    fn ep_lane_buffer_differs_from_layer_parity_across_lanes() {
        // Lane 0 ends at layer 47 (buffer 1 by ordinal 47); lane 1 layer 0 is
        // ordinal 48 -> buffer 0 = qwen4_ep_buffer_index(0, 48) here, but
        // lane 1 layer 47 (ordinal 95) -> 1, and lane 2 layer 0 (ordinal 96)
        // -> 0; with an odd layer count layer parity would repeat a buffer.
        for layers in [47usize, 48] {
            let mut repeats = 0;
            let mut last = None;
            for ordinal in 0..layers * 2 {
                let layer_parity = qwen4_ep_buffer_index(ordinal % layers, 48);
                if last == Some(layer_parity) {
                    repeats += 1;
                }
                last = Some(layer_parity);
            }
            assert_eq!(repeats > 0, layers % 2 == 1);
        }
    }
}
